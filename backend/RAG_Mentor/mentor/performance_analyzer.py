"""
Performance Analyzer

Analyzes backtest results and generates natural language explanations.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from RAG_Mentor.mentor.rag_engine import RAGEngine
from RAG_Mentor.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes backtest performance and correlates with market events.
    """
    
    def __init__(self, rag_engine: RAGEngine = None):
        """
        Initialize performance analyzer.
        
        Args:
            rag_engine: RAG engine instance
        """
        self.rag = rag_engine or RAGEngine()
    
    def analyze_backtest(
        self,
        performance_summary: Dict[str, Any],
        symbols: Optional[List[str]] = None
    ) -> str:
        """
        Generate comprehensive performance analysis.
        
        Args:
            performance_summary: Dictionary with backtest metrics:
                - total_return: Total return percentage
                - sharpe_ratio: Risk-adjusted return
                - max_drawdown: Maximum drawdown
                - win_rate: Percentage of winning trades
                - profit_factor: Gross profit / gross loss
                - total_trades: Number of trades
                - avg_win: Average winning trade
                - avg_loss: Average losing trade
            symbols: List of traded symbols
            
        Returns:
            Natural language analysis
        """
        logger.info("Analyzing backtest performance...")
        
        # Retrieve relevant context
        context = self.rag.retrieve_context_for_analysis(
            performance_summary,
            symbols
        )
        
        # Build prompt
        retrieved_context = f"""
**Trading Principles:**
{context['principles']}

**Market News Context:**
{context['news']}
"""
        
        prompt = PromptTemplates.performance_analysis(
            backtest_summary=performance_summary,
            retrieved_context=retrieved_context
        )
        
        # Generate analysis
        return self.rag.generate_with_context(prompt, temperature=0.7, max_tokens=2048)
    
    def correlate_drawdowns_with_news(
        self,
        drawdown_periods: List[Dict[str, Any]]
    ) -> str:
        """
        Correlate drawdown periods with news events.
        
        Args:
            drawdown_periods: List of dicts with:
                - start_date: Drawdown start
                - end_date: Drawdown end
                - magnitude: Drawdown percentage
                - trades: Trades during period
                
        Returns:
            Analysis of drawdown causes
        """
        if not drawdown_periods:
            return "No significant drawdown periods identified."
        
        analyses = []
        
        for i, period in enumerate(drawdown_periods):
            start_date = period['start_date']
            end_date = period['end_date']
            magnitude = period['magnitude']
            
            # Retrieve news from this period
            query = f"market volatility and events between {start_date} and {end_date}"
            news_context = self.rag.retrieve_news(
                query,
                start_date=start_date,
                end_date=end_date,
                top_k=3
            )
            
            analysis = f"""
**Drawdown Period {i+1}** ({start_date.date()} to {end_date.date()}):
- Magnitude: {magnitude:.2%}
- Market Events:
{news_context}
"""
            analyses.append(analysis)
        
        return "\n".join(analyses)
    
    def regime_performance_breakdown(
        self,
        regime_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze performance by market regime.
        
        Args:
            regime_data: Dictionary mapping regime to metrics:
                {
                    "BULLISH": {"return": 0.15, "win_rate": 0.65, "trades": 50},
                    "BEARISH": {"return": -0.08, "win_rate": 0.42, "trades": 30},
                    ...
                }
                
        Returns:
            Analysis dict with best/worst regimes and recommendations
        """
        if not regime_data:
            return {"summary": "No regime data available"}
        
        # Find best and worst regimes
        best_regime = max(regime_data.items(), key=lambda x: x[1]['return'])
        worst_regime = min(regime_data.items(), key=lambda x: x[1]['return'])
        
        analysis = {
            "best_regime": {
                "name": best_regime[0],
                "return": best_regime[1]['return'],
                "win_rate": best_regime[1].get('win_rate', 0),
                "trades": best_regime[1].get('trades', 0)
            },
            "worst_regime": {
                "name": worst_regime[0],
                "return": worst_regime[1]['return'],
                "win_rate": worst_regime[1].get('win_rate', 0),
                "trades": worst_regime[1].get('trades', 0)
            },
            "recommendation": self._generate_regime_recommendation(worst_regime[0])
        }
        
        return analysis
    
    def _generate_regime_recommendation(self, poor_regime: str) -> str:
        """Generate recommendation based on poor-performing regime"""
        recommendations = {
            "RANGING": "Consider adding an ADX filter (ADX < 20) to disable the strategy during non-trending periods.",
            "VOLATILE": "Implement volatility-adjusted position sizing. Reduce position sizes by 50% when VIX > 30 or ATR > historical average.",
            "BEARISH": "Add market filter to exit all positions when index breaks below 200-day MA. Consider shorting or inverse positions.",
            "CRISIS": "Implement protective stops or tail-risk hedges. Consider systematic hedging with put options during high volatility."
        }
        
        return recommendations.get(
            poor_regime,
            f"Review strategy logic for {poor_regime} conditions and consider regime-specific parameter adjustments."
        )
    
    def compare_to_benchmark(
        self,
        strategy_performance: Dict[str, Any],
        benchmark_performance: Dict[str, Any],
        benchmark_name: str = "SPY Buy-and-Hold"
    ) -> str:
        """
        Compare strategy to benchmark.
        
        Args:
            strategy_performance: Strategy metrics
            benchmark_performance: Benchmark metrics
            benchmark_name: Name of benchmark
            
        Returns:
            Comparison analysis
        """
        comparison = f"""
## Strategy vs {benchmark_name}

| Metric | Strategy | {benchmark_name} | Difference |
|--------|----------|------------------|------------|
| Total Return | {strategy_performance.get('total_return', 0):.2%} | {benchmark_performance.get('total_return', 0):.2%} | {(strategy_performance.get('total_return', 0) - benchmark_performance.get('total_return', 0)):.2%} |
| Sharpe Ratio | {strategy_performance.get('sharpe_ratio', 0):.2f} | {benchmark_performance.get('sharpe_ratio', 0):.2f} | {(strategy_performance.get('sharpe_ratio', 0) - benchmark_performance.get('sharpe_ratio', 0)):.2f} |
| Max Drawdown | {strategy_performance.get('max_drawdown', 0):.2%} | {benchmark_performance.get('max_drawdown', 0):.2%} | {(strategy_performance.get('max_drawdown', 0) - benchmark_performance.get('max_drawdown', 0)):.2%} |
| Win Rate | {strategy_performance.get('win_rate', 0):.2%} | N/A | - |
| Total Trades | {strategy_performance.get('total_trades', 0)} | N/A | - |

"""
        
        # Determine if strategy adds value
        excess_return = strategy_performance.get('total_return', 0) - benchmark_performance.get('total_return', 0)
        better_sharpe = strategy_performance.get('sharpe_ratio', 0) > benchmark_performance.get('sharpe_ratio', 0)
        
        if excess_return > 0 and better_sharpe:
            verdict = "✅ **Strategy outperforms** on both absolute and risk-adjusted basis."
        elif excess_return > 0:
            verdict = "⚠️ **Mixed results**: Higher returns but worse risk-adjusted performance."
        else:
            verdict = "❌ **Strategy underperforms**: Consider buy-and-hold or strategy revision."
        
        comparison += f"\n{verdict}"
        
        return comparison
