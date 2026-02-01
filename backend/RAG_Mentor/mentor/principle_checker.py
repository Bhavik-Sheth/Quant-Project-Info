"""
Principle Checker

Detects violations of trading principles in trade sequences.
"""

from typing import Dict, List, Any
import logging
from collections import defaultdict

from RAG_Mentor.mentor.rag_engine import RAGEngine
from RAG_Mentor.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class PrincipleChecker:
    """
    Analyzes trade sequences for violations of trading principles.
    """
    
    def __init__(self, rag_engine: RAGEngine = None):
        """
        Initialize principle checker.
        
        Args:
            rag_engine: RAG engine instance
        """
        self.rag = rag_engine or RAGEngine()
    
    def check_all_violations(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run all violation checks on trade sequence.
        
        Args:
            trades: List of trade dicts with keys:
                - timestamp: Trade datetime
                - symbol: Symbol traded
                - action: BUY/SELL
                - quantity: Shares
                - price: Execution price
                - pnl: Profit/loss (optional)
                - reason: Trade reason (optional)
                
        Returns:
            Dictionary with all violations found
        """
        violations = {
            "averaging_down": self.detect_averaging_down(trades),
            "stop_loss_issues": self.detect_stop_loss_violations(trades),
            "overtrading": self.detect_overtrading(trades),
            "position_sizing": self.detect_position_sizing_issues(trades),
            "summary": ""
        }
        
        # Count severity
        critical_count = sum(1 for v in violations.values() if isinstance(v, dict) and v.get('severity') == 'critical')
        moderate_count = sum(1 for v in violations.values() if isinstance(v, dict) and v.get('severity') == 'moderate')
        
        violations["summary"] = f"Found {critical_count} critical and {moderate_count} moderate violations."
        
        return violations
    
    def detect_averaging_down(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect averaging down pattern (adding to losing positions).
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        
        # Group trades by symbol
        symbol_trades = defaultdict(list)
        for trade in trades:
            symbol_trades[trade['symbol']].append(trade)
        
        # Check each symbol
        for symbol, symbol_trade_list in symbol_trades.items():
            # Sort by timestamp
            sorted_trades = sorted(symbol_trade_list, key=lambda t: t['timestamp'])
            
            # Look for pattern: BUY, price drops, BUY again
            for i in range(len(sorted_trades) - 1):
                current = sorted_trades[i]
                next_trade = sorted_trades[i + 1]
                
                if (current['action'] == 'BUY' and 
                    next_trade['action'] == 'BUY' and
                    next_trade['price'] < current['price'] * 0.95):  # 5% drop
                    
                    violations.append({
                        "symbol": symbol,
                        "first_entry": current['price'],
                        "second_entry": next_trade['price'],
                        "drop_pct": (next_trade['price'] / current['price'] - 1),
                        "dates": [current['timestamp'], next_trade['timestamp']]
                    })
        
        return {
            "found": len(violations) > 0,
            "severity": "critical" if len(violations) > 3 else "moderate" if len(violations) > 0 else "none",
            "count": len(violations),
            "examples": violations[:3],  # Top 3 examples
            "principle": "Never average down on losing positions (Jesse Livermore)"
        }
    
    def detect_stop_loss_violations(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect lack of stop losses or stops not being respected.
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        
        # Group by symbol for position tracking
        symbol_positions = defaultdict(list)
        
        for trade in trades:
            symbol = trade['symbol']
            symbol_positions[symbol].append(trade)
        
        # Check for large losses (indicating no stop loss)
        for symbol, position_trades in symbol_positions.items():
            for trade in position_trades:
                if trade.get('pnl') and trade['pnl'] < 0:
                    loss_pct = abs(trade['pnl']) / (trade['price'] * trade['quantity'])
                    
                    # Loss > 10% suggests no stop was used
                    if loss_pct > 0.10:
                        violations.append({
                            "symbol": symbol,
                            "loss_pct": loss_pct,
                            "date": trade['timestamp'],
                            "pnl": trade['pnl']
                        })
        
        return {
            "found": len(violations) > 0,
            "severity": "critical" if len(violations) > 5 else "moderate" if len(violations) > 0 else "none",
            "count": len(violations),
            "examples": violations[:3],
            "principle": "Always use stop losses (Mark Minervini)"
        }
    
    def detect_overtrading(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect overtrading (very short holding periods).
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary with violation details
        """
        if len(trades) < 2:
            return {"found": False, "severity": "none"}
        
        # Calculate holding periods
        symbol_holdings = defaultdict(list)
        
        for trade in trades:
            symbol_holdings[trade['symbol']].append(trade)
        
        short_holds = []
        
        for symbol, symbol_trades in symbol_holdings.items():
            sorted_trades = sorted(symbol_trades, key=lambda t: t['timestamp'])
            
            for i in range(len(sorted_trades) - 1):
                if sorted_trades[i]['action'] == 'BUY' and sorted_trades[i+1]['action'] == 'SELL':
                    holding_period = (sorted_trades[i+1]['timestamp'] - sorted_trades[i]['timestamp']).days
                    
                    if holding_period < 2:
                        short_holds.append({
                            "symbol": symbol,
                            "holding_days": holding_period,
                            "entry_date": sorted_trades[i]['timestamp'],
                            "exit_date": sorted_trades[i+1]['timestamp']
                        })
        
        # Calculate average holding period
        total_trades = len([t for t in trades if t['action'] == 'BUY'])
        short_hold_pct = len(short_holds) / total_trades if total_trades > 0 else 0
        
        return {
            "found": short_hold_pct > 0.3,  # >30% of trades are < 2 days
            "severity": "moderate" if short_hold_pct > 0.5 else "minor" if short_hold_pct > 0.3 else "none",
            "short_hold_count": len(short_holds),
            "short_hold_pct": short_hold_pct,
            "examples": short_holds[:3],
            "principle": "Avoid overtrading - quality over quantity"
        }
    
    def detect_position_sizing_issues(
        self,
        trades: List[Dict[str, Any]],
        account_size: float = 100000
    ) -> Dict[str, Any]:
        """
        Detect erratic or oversized positions.
        
        Args:
            trades: List of trades
            account_size: Total account size
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        
        for trade in trades:
            position_value = trade['price'] * trade['quantity']
            position_pct = position_value / account_size
            
            # Flag if position > 20% of account
            if position_pct > 0.20:
                violations.append({
                    "symbol": trade['symbol'],
                    "position_pct": position_pct,
                    "position_value": position_value,
                    "date": trade['timestamp']
                })
        
        return {
            "found": len(violations) > 0,
            "severity": "critical" if len(violations) > 5 else "moderate" if len(violations) > 0 else "none",
            "count": len(violations),
            "examples": violations[:3],
            "principle": "Never risk more than 20% of capital in a single position"
        }
    
    def generate_violation_report(
        self,
        trades: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive violation report with LLM analysis.
        
        Args:
            trades: List of trades
            
        Returns:
            Natural language violation report
        """
        # Run all checks
        violations = self.check_all_violations(trades)
        
        # Retrieve relevant principles
        retrieved_principles = self.rag.retrieve_principles(
            "Trading discipline, risk management, position sizing",
            top_k=8,
            category_filter="risk_management"
        )
        
        # Build prompt
        prompt = PromptTemplates.principle_violation_check(
            trade_sequence=trades[:50],  # Limit to first 50 for context size
            retrieved_principles=retrieved_principles
        )
        
        # Generate report
        llm_analysis = self.rag.generate_with_context(prompt, temperature=0.6, max_tokens=1500)
        
        # Combine with programmatic detection
        full_report = f"""# Trading Principle Violation Report

## Automated Detection Summary
{violations['summary']}

### 1. Averaging Down
- Violations found: {violations['averaging_down']['count']}
- Severity: {violations['averaging_down']['severity']}

### 2. Stop Loss Issues
- Violations found: {violations['stop_loss_issues']['count']}
- Severity: {violations['stop_loss_issues']['severity']}

### 3. Overtrading
- Short holding periods: {violations['overtrading'].get('short_hold_count', 0)}
- Severity: {violations['overtrading']['severity']}

### 4. Position Sizing
- Oversized positions: {violations['position_sizing']['count']}
- Severity: {violations['position_sizing']['severity']}

---

## LLM Analysis

{llm_analysis}
"""
        
        return full_report
