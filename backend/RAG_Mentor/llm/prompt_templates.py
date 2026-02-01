"""
Prompt Templates for Trading Mentor

Specialized prompts for different analysis tasks.
"""

from typing import Dict, List, Any


class PromptTemplates:
    """Collection of prompt templates for trading analysis"""
    
    @staticmethod
    def performance_analysis(
        backtest_summary: Dict[str, Any],
        retrieved_context: str
    ) -> str:
        """
        Prompt for comprehensive performance analysis.
        
        Args:
            backtest_summary: Dictionary with performance metrics
            retrieved_context: Retrieved trading principles and news
            
        Returns:
            Formatted prompt
        """
        return f"""You are an expert trading mentor analyzing backtest results.

**Backtest Performance Summary:**
{PromptTemplates._format_metrics(backtest_summary)}

**Relevant Context (Retrieved from Knowledge Base):**
{retrieved_context}

**Your Task:**
Provide a comprehensive analysis covering:

1. **Performance Explanation**: 
   - What drove the returns (positive and negative)?
   - Correlate drawdown periods with market events from the context
   - Identify which market regimes the strategy performed well/poorly in

2. **Risk Assessment**:
   - Analyze the risk-adjusted returns (Sharpe ratio, max drawdown)
   - Evaluate the win rate and profit factor
   - Comment on position sizing and capital efficiency

3. **Market Context**:
   - Connect performance to retrieved news events
   - Explain how macro conditions affected the strategy
   - Identify periods of regime mismatch

Provide your analysis in clear, structured sections.
"""
    
    @staticmethod
    def principle_violation_check(
        trade_sequence: List[Dict[str, Any]],
        retrieved_principles: str
    ) -> str:
        """
        Prompt for detecting violations of trading principles.
        
        Args:
            trade_sequence: List of trades
            retrieved_principles: Retrieved trading wisdom
            
        Returns:
            Formatted prompt
        """
        trades_summary = PromptTemplates._format_trades(trade_sequence)
        
        return f"""You are a trading discipline auditor. Analyze the following trade sequence against established trading principles.

**Trade Sequence:**
{trades_summary}

**Trading Principles to Check:**
{retrieved_principles}

**Detection Tasks:**

1. **Averaging Down**: Did the strategy add to losing positions repeatedly?
2. **Stop-Loss Discipline**: Were stop losses used? Were they respected?
3. **Overtrading**: Are holding periods extremely short (< 2 days)?
4. **Position Sizing**: Was position sizing consistent or erratic?
5. **Trend Alignment**: Did trades go against the major trend?
6. **Emotional Trading**: Evidence of revenge trading or panic exits?

For each violation found, provide:
- Specific trade examples
- Which principle was violated
- Severity (minor/moderate/critical)
- Potential loss impact

Be thorough but concise.
"""
    
    @staticmethod
    def improvement_suggestions(
        performance_summary: Dict[str, Any],
        regime_breakdown: Dict[str, Any],
        retrieved_insights: str
    ) -> str:
        """
        Prompt for generating improvement suggestions.
        
        Args:
            performance_summary: Overall performance metrics
            regime_breakdown: Performance by market regime
            retrieved_insights: Retrieved trading knowledge
            
        Returns:
            Formatted prompt
        """
        return f"""You are a quantitative strategy consultant. Based on the backtest analysis, suggest concrete improvements.

**Performance Summary:**
{PromptTemplates._format_metrics(performance_summary)}

**Regime-Wise Performance:**
{PromptTemplates._format_metrics(regime_breakdown)}

**Available Knowledge:**
{retrieved_insights}

**Provide Specific, Actionable Suggestions:**

1. **Regime Filters**: 
   - Should the strategy use ADX/ATR to detect unsuitable market conditions?
   - Specific threshold recommendations based on the data

2. **Risk Management**:
   - Volatility-adjusted position sizing recommendations
   - Stop-loss placement improvements
   - Portfolio heat limits

3. **Entry/Exit Optimization**:
   - Additional indicators to filter false signals
   - Exit timing improvements
   - Take-profit strategies

4. **Hedging**:
   - Should options be used during high volatility periods?
   - Protective put suggestions based on predicted volatility

5. **Parameter Tuning**:
   - Which parameters need adjustment?
   - Suggested ranges based on regime performance

Format each suggestion with:
- **Suggestion**: What to change
- **Rationale**: Why it should help (backed by data)
- **Implementation**: How to implement it
- **Expected Impact**: Quantified if possible
"""
    
    @staticmethod
    def conversational_query(
        conversation_history: List[Dict[str, str]],
        current_query: str,
        context: str
    ) -> str:
        """
        Prompt for conversational follow-up questions.
        
        Args:
            conversation_history: Previous messages
            current_query: Current user question
            context: Retrieved context for query
            
        Returns:
            Formatted prompt
        """
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-5:]  # Last 5 messages
        ])
        
        return f"""You are a Trading Mentor assistant helping a user analyze their strategy.

**Conversation History:**
{history_text}

**Current Question:**
{current_query}

**Relevant Context:**
{context}

Provide a helpful, concise answer. If the question asks for specific trades, format them as a table. If comparing strategies, provide side-by-side metrics. Be data-driven and precise.
"""
    
    @staticmethod
    def trade_analysis_query(
        query: str,
        trades: List[Dict[str, Any]],
        news_context: str
    ) -> str:
        """
        Prompt for analyzing specific trades.
        
        Args:
            query: User query about trades
            trades: Filtered trade records
            news_context: News from the period
            
        Returns:
            Formatted prompt
        """
        trades_text = PromptTemplates._format_trades(trades)
        
        return f"""Analyze the following trades based on the user's question.

**User Query:**
{query}

**Trades:**
{trades_text}

**Market News During This Period:**
{news_context}

Provide insights that directly answer the user's question. Include relevant statistics and correlations.
"""
    
    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary for display"""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower():
                    lines.append(f"- {key}: {value:.2%}")
                else:
                    lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_trades(trades: List[Dict[str, Any]], max_trades: int = 20) -> str:
        """Format trade list for display"""
        if not trades:
            return "No trades available"
        
        lines = ["Trade#\tDate\t\tSymbol\tAction\tQuantity\tPrice\tP&L"]
        lines.append("-" * 70)
        
        for i, trade in enumerate(trades[:max_trades]):
            lines.append(
                f"{i+1}\t{trade.get('timestamp', 'N/A')}\t"
                f"{trade.get('symbol', 'N/A')}\t{trade.get('action', 'N/A')}\t"
                f"{trade.get('quantity', 0)}\t{trade.get('price', 0):.2f}\t"
                f"{trade.get('pnl', 0):.2f}"
            )
        
        if len(trades) > max_trades:
            lines.append(f"... and {len(trades) - max_trades} more trades")
        
        return "\n".join(lines)
