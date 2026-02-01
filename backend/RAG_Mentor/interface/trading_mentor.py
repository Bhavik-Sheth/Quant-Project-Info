"""
Trading Mentor

Main interface for the RAG-based Trading Mentor system.
Provides comprehensive backtest analysis, principle checking, improvement suggestions,
and conversational Q&A capabilities.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd

from RAG_Mentor.mentor.rag_engine import RAGEngine
from RAG_Mentor.mentor.performance_analyzer import PerformanceAnalyzer
from RAG_Mentor.mentor.principle_checker import PrincipleChecker
from RAG_Mentor.mentor.improvement_engine import ImprovementEngine
from RAG_Mentor.interface.conversation_manager import ConversationManager
from RAG_Mentor.llm.prompt_templates import PromptTemplates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingMentor:
    """
    Main Trading Mentor interface combining RAG, performance analysis,
    principle checking, and conversational AI.
    """
    
    def __init__(self):
        """Initialize Trading Mentor with all components"""
        logger.info("Initializing Trading Mentor...")
        
        # Core components
        self.rag = RAGEngine()
        self.performance_analyzer = PerformanceAnalyzer(self.rag)
        self.principle_checker = PrincipleChecker(self.rag)
        self.improvement_engine = ImprovementEngine(self.rag)
        self.conversation = ConversationManager()
        
        logger.info("Trading Mentor ready!")
    
    def analyze_backtest(
        self,
        performance_summary: Dict[str, Any],
        trades: List[Dict[str, Any]],
        symbols: Optional[List[str]] = None,
        regime_breakdown: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive backtest analysis.
        
        This is the main entry point for analyzing a completed backtest.
        
        Args:
            performance_summary: Dictionary with overall metrics:
                - total_return: Overall return
                - sharpe_ratio: Risk-adjusted return
                - max_drawdown: Maximum drawdown
                - win_rate: Percentage of winning trades
                - profit_factor: Gross profit / gross loss
                - total_trades: Number of trades
            trades: List of individual trades
            symbols: List of traded symbols
            regime_breakdown: Performance by market regime (optional)
            
        Returns:
            Dictionary with analysis sections:
                - performance_analysis: Overall performance explanation
                - violation_report: Trading principle violations
                - improvement_suggestions: Concrete recommendations
                - summary: Executive summary
        """
        logger.info(f"Analyzing backtest with {len(trades)} trades...")
        
        # Store in conversation context
        self.conversation.set_context('last_backtest', {
            'performance': performance_summary,
            'trades': trades,
            'symbols': symbols,
            'timestamp': datetime.now()
        })
        
        # 1. Performance analysis
        logger.info("Step 1/3: Analyzing performance...")
        performance_analysis = self.performance_analyzer.analyze_backtest(
            performance_summary,
            symbols
        )
        
        # 2. Principle violation check
        logger.info("Step 2/3: Checking for principle violations...")
        violation_report = self.principle_checker.generate_violation_report(trades)
        
        # 3. Improvement suggestions
        logger.info("Step 3/3: Generating improvement suggestions...")
        improvement_suggestions = self.improvement_engine.generate_improvements(
            performance_summary,
            regime_breakdown,
            trade_analysis=None
        )
        
        # 4. Executive summary
        summary = self._generate_executive_summary(
            performance_summary,
            len(trades)
        )
        
        # Add to conversation history
        self.conversation.add_message("user", "Analyze my backtest results")
        self.conversation.add_message("assistant", summary)
        
        logger.info("Backtest analysis complete!")
        
        return {
            "summary": summary,
            "performance_analysis": performance_analysis,
            "violation_report": violation_report,
            "improvement_suggestions": improvement_suggestions
        }
    
    def ask_question(self, question: str) -> str:
        """
        Ask a follow-up question about the backtest.
        
        Args:
            question: User question (e.g., "Show me all trades during March 2020")
            
        Returns:
            Natural language answer
        """
        # Classify intent
        intent = self.conversation.classify_intent(question)
        logger.info(f"Query intent: {intent}")
        
        # Get conversation history
        history = self.conversation.get_recent_history(5)
        
        # Get context
        backtest_context = self.conversation.get_context('last_backtest')
        
        if not backtest_context:
            return "Please run `analyze_backtest()` first to establish context."
        
        # Handle specific intents
        if intent == 'show_trades':
            return self._handle_show_trades(question, backtest_context)
        elif intent == 'compare':
            return self._handle_comparison(question, backtest_context)
        else:
            # General conversational query
            return self._handle_general_query(question, history, backtest_context)
    
    def _handle_show_trades(
        self,
        query: str,
        backtest_context: Dict[str, Any]
    ) -> str:
        """Handle 'show trades' queries"""
        trades = backtest_context.get('trades', [])
        
        # Parse query for filtering (simple keyword matching)
        filtered_trades = trades
        
        # Date filtering
        if 'march' in query.lower() and '2020' in query.lower():
            filtered_trades = [
                t for t in trades
                if isinstance(t.get('timestamp'), datetime) and
                t['timestamp'].year == 2020 and t['timestamp'].month == 3
            ]
        
        # Symbol filtering
        symbols = backtest_context.get('symbols', [])
        for symbol in symbols:
            if symbol.lower() in query.lower():
                filtered_trades = [
                    t for t in trades
                    if t.get('symbol', '').lower() == symbol.lower()
                ]
                break
        
        # Format trades as table
        if not filtered_trades:
            return f"No trades found matching: {query}"
        
        # Get news context for the period if trades found
        if filtered_trades:
            start_date = min(t['timestamp'] for t in filtered_trades)
            end_date = max(t['timestamp'] for t in filtered_trades)
            
            news_context = self.rag.retrieve_news(
                f"market events during {start_date.date()} to {end_date.date()}",
                start_date=start_date,
                end_date=end_date,
                top_k=3
            )
            
            # Generate analysis
            prompt = PromptTemplates.trade_analysis_query(
                query=query,
                trades=filtered_trades[:50],  # Limit to 50 for context
                news_context=news_context
            )
            
            return self.rag.generate_with_context(prompt, temperature=0.6)
        
        return "No trades found."
    
    def _handle_comparison(
        self,
        query: str,
        backtest_context: Dict[str, Any]
    ) -> str:
        """Handle comparison queries"""
        performance = backtest_context.get('performance', {})
        
        # Create simple buy-and-hold benchmark
        benchmark = {
            'total_return': 0.10,  # Assume 10% SPY annual return
            'sharpe_ratio': 0.8,
            'max_drawdown': 0.20
        }
        
        comparison = self.performance_analyzer.compare_to_benchmark(
            strategy_performance=performance,
            benchmark_performance=benchmark,
            benchmark_name="SPY Buy-and-Hold"
        )
        
        return comparison
    
    def _handle_general_query(
        self,
        query: str,
        history: List[Dict[str, str]],
        backtest_context: Dict[str, Any]
    ) -> str:
        """Handle general conversational queries"""
        # Build context from backtest
        context_summary = f"""
Current backtest context:
- Total trades: {len(backtest_context.get('trades', []))}
- Symbols: {', '.join(backtest_context.get('symbols', []))}
- Total return: {backtest_context['performance'].get('total_return', 0):.2%}
- Sharpe ratio: {backtest_context['performance'].get('sharpe_ratio', 0):.2f}
"""
        
        # Retrieve relevant context based on query
        retrieved_context = self.rag.retrieve_and_generate(
            query=query,
            context_type="both",
            additional_context=context_summary,
            temperature=0.7
        )
        
        # Add to conversation
        self.conversation.add_message("user", query)
        self.conversation.add_message("assistant", retrieved_context)
        
        return retrieved_context
    
    def _generate_executive_summary(
        self,
        performance: Dict[str, Any],
        trade_count: int
    ) -> str:
        """Generate executive summary"""
        summary = f"""# Trading Strategy Analysis Summary

**Performance Overview:**
- Total Return: {performance.get('total_return', 0):.2%}
- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
- Maximum Drawdown: {performance.get('max_drawdown', 0):.2%}
- Win Rate: {performance.get('win_rate', 0):.2%}
- Total Trades: {trade_count}

**Quick Assessment:**
"""
        
        # Simple scoring
        score = 0
        if performance.get('sharpe_ratio', 0) > 1.5:
            score += 2
        elif performance.get('sharpe_ratio', 0) > 1.0:
            score += 1
        
        if performance.get('max_drawdown', 0) < 0.15:
            score += 2
        elif performance.get('max_drawdown', 0) < 0.25:
            score += 1
        
        if performance.get('win_rate', 0) > 0.6:
            score += 2
        elif performance.get('win_rate', 0) > 0.5:
            score += 1
        
        if score >= 5:
            verdict = "✅ **Strong Strategy** - Shows consistent edge with good risk management"
        elif score >= 3:
            verdict = "⚠️ **Promising but Needs Work** - Has potential, requires optimization"
        else:
            verdict = "❌ **Needs Significant Improvement** - Consider major revisions"
        
        summary += verdict + "\n\n"
        summary += "See detailed analysis, principle violations, and improvement suggestions below."
        
        return summary
    
    def reset_conversation(self) -> None:
        """Reset conversation state"""
        self.conversation.reset_session()
        logger.info("Conversation reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation.conversation_history
