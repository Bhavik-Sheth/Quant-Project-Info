"""
RAG Engine

Core retrieval-augmented generation engine combining vector search with LLM reasoning.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from RAG_Mentor.vector_db.chroma_manager import ChromaManager
from RAG_Mentor.llm.llm_client import get_llm_client
from RAG_Mentor.config import Config

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine that retrieves relevant context and generates insights.
    """
    
    def __init__(self, chroma_manager: ChromaManager = None):
        """
        Initialize RAG engine.
        
        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma = chroma_manager or ChromaManager()
        self.llm = get_llm_client()
        logger.info(f"RAG Engine initialized with LLM: {self.llm.get_active_provider()}")
    
    def retrieve_principles(
        self,
        query: str,
        top_k: int = None,
        category_filter: Optional[str] = None
    ) -> str:
        """
        Retrieve relevant trading principles.
        
        Args:
            query: Query describing the situation
            top_k: Number of principles to retrieve
            category_filter: Optional category filter
            
        Returns:
            Formatted string of retrieved principles
        """
        top_k = top_k or Config.TOP_K_RESULTS
        
        results = self.chroma.search_principles(
            query=query,
            top_k=top_k,
            category_filter=category_filter
        )
        
        if not results.get('documents') or not results['documents'][0]:
            return "No relevant principles found."
        
        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted.append(f"**Principle {i+1}** (Author: {meta.get('author', 'Unknown')}):")
            formatted.append(f"{doc}")
            formatted.append(f"(Relevance: {1 - dist:.2f})\n")
        
        return "\n".join(formatted)
    
    def retrieve_news(
        self,
        query: str,
        top_k: int = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve relevant news articles.
        
        Args:
            query: Query describing the event/period
            top_k: Number of articles to retrieve
            start_date: Filter start date
            end_date: Filter end date
            symbols: Filter by symbols
            
        Returns:
            Formatted string of news articles
        """
        top_k = top_k or Config.TOP_K_RESULTS
        
        results = self.chroma.search_news(
            query=query,
            top_k=top_k,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        if not results.get('documents') or not results['documents'][0]:
            return "No relevant news found."
        
        formatted = []
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        )):
            formatted.append(f"**News {i+1}** ({meta.get('timestamp', 'Unknown date')}):")
            formatted.append(f"{meta.get('headline', 'No headline')}")
            formatted.append(f"Related symbols: {meta.get('symbols', 'N/A')}")
            formatted.append(f"Source: {meta.get('source', 'Unknown')}\n")
        
        return "\n".join(formatted)
    
    def retrieve_context_for_analysis(
        self,
        performance_summary: Dict[str, Any],
        symbols: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Retrieve comprehensive context for backtest analysis.
        
        Args:
            performance_summary: Backtest performance metrics
            symbols: Traded symbols
            
        Returns:
            Dictionary with retrieved context
        """
        context = {}
        
        # Build query from performance characteristics
        query_components = []
        
        # Add performance characteristics
        if performance_summary.get('sharpe_ratio', 0) < 1.0:
            query_components.append("low risk-adjusted returns")
        
        if performance_summary.get('max_drawdown', 0) > 0.15:
            query_components.append("large drawdown")
        
        if performance_summary.get('win_rate', 0) < 0.5:
            query_components.append("low win rate")
        
        query = " ".join(query_components) if query_components else "trading strategy analysis"
        
        # Retrieve principles
        logger.info(f"Retrieving principles for: {query}")
        context['principles'] = self.retrieve_principles(query, top_k=5)
        
        # Retrieve relevant news
        if symbols:
            news_query = f"market conditions affecting {', '.join(symbols)}"
        else:
            news_query = "market volatility and conditions"
        
        logger.info(f"Retrieving news for: {news_query}")
        context['news'] = self.retrieve_news(news_query, top_k=5, symbols=symbols)
        
        return context
    
    def generate_with_context(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate LLM response.
        
        Args:
            prompt: Full prompt with context
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Generated text
        """
        return self.llm.generate(prompt, temperature, max_tokens)
    
    def retrieve_and_generate(
        self,
        query: str,
        context_type: str = "principles",
        additional_context: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Retrieve context and generate response in one call.
        
        Args:
            query: User query
            context_type: Type of context ("principles", "news", or "both")
            additional_context: Additional context to include
            temperature: LLM temperature
            
        Returns:
            Generated response with retrieved context
        """
        # Retrieve context
        retrieved_context = []
        
        if context_type in ["principles", "both"]:
            retrieved_context.append(self.retrieve_principles(query))
        
        if context_type in ["news", "both"]:
            retrieved_context.append(self.retrieve_news(query))
        
        # Build full prompt
        prompt = f"""Based on the following context, answer the query.

**Retrieved Context:**
{chr(10).join(retrieved_context)}

{f'**Additional Context:**{chr(10)}{additional_context}{chr(10)}' if additional_context else ''}

**Query:**
{query}

**Answer:**
"""
        
        return self.generate_with_context(prompt, temperature=temperature)
