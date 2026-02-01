"""
Knowledge Loader

Ingests trading principles and news into ChromaDB.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from RAG_Mentor.vector_db.chroma_manager import ChromaManager
from RAG_Mentor.config import Config

logger = logging.getLogger(__name__)


class KnowledgeLoader:
    """Loads trading knowledge into the vector database"""
    
    def __init__(self, chroma_manager: ChromaManager = None):
        """
        Initialize knowledge loader.
        
        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma = chroma_manager or ChromaManager()
        self.knowledge_dir = Config.RAG_DIR / "knowledge"
    
    def load_trading_principles(self, filepath: str = None) -> int:
        """
        Load trading principles from JSON file.
        
        Args:
            filepath: Path to principles JSON file
            
        Returns:
            Number of principles loaded
        """
        filepath = filepath or (self.knowledge_dir / "trading_principles.json")
        
        logger.info(f"Loading trading principles from {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        principles = data.get('principles', [])
        count = 0
        
        for p in principles:
            # Format explanation with examples
            explanation = p['explanation']
            if p.get('examples'):
                explanation += "\n\nExamples:\n"
                explanation += "\n".join(f"- {ex}" for ex in p['examples'])
            
            self.chroma.add_trading_principle(
                principle=p['principle'],
                explanation=explanation,
                author=p['author'],
                category=p['category']
            )
            count += 1
        
        logger.info(f"Loaded {count} trading principles")
        return count
    
    def add_news_from_dict(
        self,
        news_articles: List[Dict[str, Any]]
    ) -> int:
        """
        Add news articles from a list of dictionaries.
        
        Args:
            news_articles: List of news article dicts with keys:
                - headline: Article headline
                - summary: Article summary
                - timestamp: Publication datetime
                - symbols: List of related symbols
                - source: News source (optional)
                
        Returns:
            Number of articles added
        """
        count = 0
        
        for article in news_articles:
            timestamp = article.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            self.chroma.add_news_article(
                headline=article['headline'],
                summary=article.get('summary', ''),
                timestamp=timestamp,
                symbols=article.get('symbols', []),
                source=article.get('source', 'Unknown')
            )
            count += 1
        
        logger.info(f"Added {count} news articles")
        return count
    
    def load_sample_news(self) -> int:
        """
        Load sample news articles for demonstration.
        
        Returns:
            Number of articles loaded
        """
        sample_news = [
            {
                "headline": "Federal Reserve Raises Interest Rates by 75 Basis Points",
                "summary": "The Federal Reserve announced an aggressive 75 basis point rate hike to combat inflation, the largest increase since 1994. Markets initially sold off on the news.",
                "timestamp": datetime(2022, 6, 15, 14, 0, 0),
                "symbols": ["SPY", "QQQ", "^VIX"],
                "source": "Federal Reserve"
            },
            {
                "headline": "Tech Stocks Plunge as Earnings Miss Expectations",
                "summary": "Major technology companies  reported weak quarterly earnings, citing slowing growth and currency headwinds. NASDAQ dropped 3.5% in heavy volume.",
                "timestamp": datetime(2022, 10, 27, 16, 30, 0),
                "symbols": ["AAPL", "MSFT", "GOOGL", "META", "QQQ"],
                "source": "Market Watch"
            },
            {
                "headline": "VIX Surges Above 30 as Market Uncertainty Increases",
                "summary": "The CBOE Volatility Index spiked above 30, indicating elevated fear in the market. Options traders are paying premiums for downside protection.",
                "timestamp": datetime(2022, 9, 13, 9, 30, 0),
                "symbols": ["^VIX", "SPY", "SPX"],
                "source": "CBOE"
            },
            {
                "headline": "Oil Prices Spike on Supply Chain Disruptions",
                "summary": "Crude oil surged 8% following geopolitical tensions and supply disruptions. Energy sector stocks rallied while broader market declined.",
                "timestamp": datetime(2022, 3, 7, 10, 15, 0),
                "symbols": ["USO", "XLE", "CVX", "XOM"],
                "source": "Bloomberg"
            },
            {
                "headline": "Banking Sector Stress Test Results Released",
                "summary": "Federal Reserve released annual stress test results showing banks well-capitalized to withstand severe recession scenario. Financial stocks gained 2%.",
                "timestamp": datetime(2023, 6, 28, 16, 30, 0),
                "symbols": ["JPM", "BAC", "WFC", "XLF"],
                "source": "Federal Reserve"
            },
            {
                "headline": "Strong Jobs Report Surprises Economists",
                "summary": "Non-farm payrolls added 528,000 jobs, far exceeding expectations of 250,000. Unemployment dropped to 3.5%, raising concerns about Fed tightening.",
                "timestamp": datetime(2022, 8, 5, 8, 30, 0),
                "symbols": ["SPY", "QQQ", "DIA"],
                "source": "Bureau of Labor Statistics"
            },
            {
                "headline": "Inflation Data Comes in Hotter Than Expected",
                "summary": "CPI rose 8.6% year-over-year, the highest in 40 years. Core inflation also accelerated. Markets pricing in more aggressive Fed action.",
                "timestamp": datetime(2022, 6, 10, 8, 30, 0),
                "symbols": ["SPY", "TLT", "GLD"],
                "source": "Bureau of Labor Statistics"
            },
            {
                "headline": "Major Bank Collapse Triggers Systemic Fears",
                "summary": "Silicon Valley Bank failed in largest bank collapse since 2008, triggering contagion fears across regional banking sector. Financial stocks plunged.",
                "timestamp": datetime(2023, 3, 10, 10, 0, 0),
                "symbols": ["SIVB", "XLF", "KRE", "SPY"],
                "source": "FDIC"
            }
        ]
        
        return self.add_news_from_dict(sample_news)
    
    def initialize_knowledge_base(self, include_sample_news: bool = True) -> Dict[str, int]:
        """
        Initialize complete knowledge base.
        
        Args:
            include_sample_news: Whether to add sample news articles
            
        Returns:
            Dictionary with counts of loaded items
        """
        logger.info("Initializing knowledge base...")
        
        results = {}
        
        # Load trading principles
        results['principles'] = self.load_trading_principles()
        
        # Load sample news if requested
        if include_sample_news:
            results['news'] = self.load_sample_news()
        else:
            results['news'] = 0
        
        logger.info(f"Knowledge base initialized: {results}")
        return results
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with collection counts
        """
        return self.chroma.get_collection_stats()
