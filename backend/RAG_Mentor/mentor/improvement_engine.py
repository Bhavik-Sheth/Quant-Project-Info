"""
Improvement Engine

Generates concrete, actionable improvement suggestions for trading strategies.
"""

from typing import Dict, List, Any, Optional
import logging

from RAG_Mentor.mentor.rag_engine import RAGEngine
from RAG_Mentor.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class ImprovementEngine:
    """
    Generates data-driven improvement suggestions for trading strategies.
    """
    
    def __init__(self, rag_engine: RAGEngine = None):
        """
        Initialize improvement engine.
        
        Args:
            rag_engine: RAG engine instance
        """
        self.rag = rag_engine or RAGEngine()
    
    def generate_improvements(
        self,
        performance_summary: Dict[str, Any],
        regime_breakdown: Optional[Dict[str, Any]] = None,
        trade_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive improvement suggestions.
        
        Args:
            performance_summary: Overall backtest metrics
            regime_breakdown: Performance by market regime
            trade_analysis: Trade-level statistics
            
        Returns:
            Natural language improvement suggestions
        """
        logger.info("Generating improvement suggestions...")
        
        # Retrieve relevant trading insights
        query = self._build_improvement_query(performance_summary, regime_breakdown)
        retrieved_insights = self.rag.retrieve_principles(query, top_k=8)
        
        # Build prompt
        prompt = PromptTemplates.improvement_suggestions(
            performance_summary=performance_summary,
            regime_breakdown=regime_breakdown or {},
            retrieved_insights=retrieved_insights
        )
        
        # Generate suggestions
        llm_suggestions = self.rag.generate_with_context(prompt, temperature=0.7, max_tokens=2500)
        
        # Add programmatic suggestions
        programmatic_suggestions = self._generate_programmatic_suggestions(
            performance_summary,
            regime_breakdown
        )
        
        # Combine
        full_report = f"""# Strategy Improvement Recommendations

## AI-Generated Insights

{llm_suggestions}

---

## Additional Technical Recommendations

{programmatic_suggestions}
"""
        
        return full_report
    
    def _build_improvement_query(
        self,
        performance: Dict[str, Any],
        regime_breakdown: Optional[Dict[str, Any]]
    ) -> str:
        """Build query to retrieve relevant improvement principles"""
        issues = []
        
        if performance.get('sharpe_ratio', 0) < 1.0:
            issues.append("improving risk-adjusted returns")
        
        if performance.get('max_drawdown', 0) > 0.20:
            issues.append("reducing drawdowns")
        
        if performance.get('win_rate', 0) < 0.5:
            issues.append("increasing win rate")
        
        if regime_breakdown:
            poor_regimes = [r for r, data in regime_breakdown.items() 
                           if data.get('return', 0) < 0]
            if poor_regimes:
                issues.append(f"handling {', '.join(poor_regimes)} markets")
        
        return " ".join(issues) if issues else "strategy optimization"
    
    def _generate_programmatic_suggestions(
        self,
        performance: Dict[str, Any],
        regime_breakdown: Optional[Dict[str, Any]]
    ) -> str:
        """Generate rule-based suggestions"""
        suggestions = []
        
        # Sharpe ratio suggestions
        if performance.get('sharpe_ratio', 0) < 1.0:
            suggestions.append("""
### 📊 Improve Sharpe Ratio
**Issue**: Sharpe ratio < 1.0 indicates poor risk-adjusted returns.

**Suggestions**:
1. **Volatility-Adjusted Position Sizing**:
   - Reduce position size during high volatility (VIX > 30)
   - Formula: `position_size = base_size * (avg_volatility / current_volatility)`
   
2. **Stricter Entry Filters**:
   - Add trend confirmation (price > 50-day MA)
   - Require RSI between 40-60 (avoid extremes)
   
3. **Improved Exit Logic**:
   - Implement trailing stops to capture extended moves
   - Take partial profits at technical resistance levels
""")
        
        # Drawdown suggestions
        if performance.get('max_drawdown', 0) > 0.15:
            suggestions.append("""
### 🛡️ Reduce Maximum Drawdown
**Issue**: Max drawdown > 15% suggests inadequate risk management.

**Suggestions**:
1. **Portfolio Heat Limits**:
   - Limit total portfolio risk to 6% at any time
   - Formula: `sum(position_risk)` should not exceed 0.06
   
2. **Correlation Management**:
   - Avoid holding multiple positions in the same sector
   - Max 30% of capital in any one sector
   
3. **Drawdown Circuit Breaker**:
   - If portfolio drawdown > 10%, reduce all positions by 50%
   - Exit remaining positions if drawdown > 15%
   - Resume normal sizing after 5-day cooling period
""")
        
        # Win rate suggestions
        if performance.get('win_rate', 0) < 0.5:
            suggestions.append("""
### 🎯 Increase Win Rate
**Issue**: Win rate < 50% requires higher profit factor to be profitable.

**Suggestions**:
1. **Stricter Stock Selection**:
   - Only trade stocks with relative strength > 80
   - Require stocks within 15% of 52-week highs
   - Minimum average volume > 1M shares
   
2. **Better Entry Timing**:
   - Wait for pullbacks to 21-EMA in uptrends
   - Enter on volume confirmation (> 1.5x avg volume)
   - Avoid buying on up-days (wait for consolidation)
   
3. **Pattern Recognition**:
   - Only trade cup-with-handle or flat-base breakouts
   - Require minimum 4-week base formation
   - Avoid choppy or volatile bases
""")
        
        # Regime-specific suggestions
        if regime_breakdown:
            worst_regime = min(regime_breakdown.items(), key=lambda x: x[1].get('return', 0))
            
            regime_suggestions = {
                "RANGING": """
### 📉 Improve Performance in Ranging Markets
**Issue**: Strategy underperforms when market lacks trend.

**Suggestions**:
1. **Add ADX Filter**:
   - Disable strategy when ADX < 25 (weak trend)
   - Reduce position size by 50% when ADX < 30
   
2. **Mean Reversion Tactics**:
   - In ranging markets (ADX < 20), switch to mean reversion
   - Buy RSI < 30, sell RSI > 70
   - Use Bollinger Bands for entry/exit signals
""",
                "VOLATILE": """
### 🌪️ Handle High Volatility Better
**Issue**: Strategy loses money during volatile periods.

**Suggestions**:
1. **Volatility-Adjusted Stops**:
   - Widen stops during high volatility: `stop = entry - (2 * ATR)`
   - Use percentage stops when VIX > 30: 7-8% instead of 5%
   
2. **Position Size Scaling**:
   - Reduce to 50% size when VIX > 25
   - Reduce to 25% size when VIX > 35
   
3. **Options Hedging**:
   - Buy protective puts when VIX percentile < 20 (cheap insurance)
   - Sell covered calls when VIX > 30 to collect premium
""",
                "BEARISH": """
### 🐻 Adapt to Bear Markets
**Issue**: Strategy fails during market downtrends.

**Suggestions**:
1. **Market Regime Filter**:
   - Exit all longs when SPY closes below 200-day MA
   - Reduce position count by 50% when SPY < 50-day MA
   
2. **Short Bias**:
   - Consider inverse positions in downtrends
   - Short stocks breaking down from Stage 3 distribution
   
3. **Defensive Positioning**:
   - Shift to defensive sectors (Consumer Staples, Healthcare)
   - Increase cash allocation to 50% or higher
"""
            }
            
            if worst_regime[0] in regime_suggestions:
                suggestions.append(regime_suggestions[worst_regime[0]])
        
        # Options suggestions if volatility data available
        if performance.get('avg_volatility'):
            suggestions.append("""
### 📈 Leverage Options Strategies
**Opportunity**: Volatility forecasting could enable options trading.

**Suggestions**:
1. **Protective Puts**:
   - When volatility forecast increases >20%, buy protective puts
   - Use 5-10% OTM puts expiring in 30-45 days
   - Cost: ~1-2% of portfolio for insurance
   
2. **Volatility Arbitrage**:
   - When implied volatility > forecasted realized volatility + threshold:
     - Sell credit spreads or iron condors
   - When implied volatility < forecasted realized volatility:
     - Buy straddles/strangles before expected moves
   
3. **Covered Calls**:
   - Sell calls against existing long positions when:
     - IV percentile > 70 (expensive premiums)
     - Stock approaching resistance levels
""")
        
        return "\n".join(suggestions) if suggestions else "No specific programmatic suggestions at this time."
    
    def suggest_regime_filter(self, regime_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Suggest specific regime filters based on performance.
        
        Args:
            regime_data: Performance by regime
            
        Returns:
            Dictionary with filter suggestions
        """
        suggestions = {}
        
        for regime, metrics in regime_data.items():
            if metrics.get('return', 0) < -0.05:  # Lost >5% in this regime
                if regime == "RANGING":
                    suggestions[regime] = "Add ADX filter: Only trade when ADX > 25"
                elif regime == "VOLATILE":
                    suggestions[regime] = "Scale position size by inverse volatility"
                elif regime == "BEARISH":
                    suggestions[regime] = "Exit when S&P 500 < 200-day MA"
        
        return suggestions
