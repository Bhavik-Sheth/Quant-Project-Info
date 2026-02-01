"""
Risk Manager Module

Provides global risk gates and portfolio constraint checks.
Integrates with RiskMonitoringAgent from AI_Agents for comprehensive risk management.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add parent directory to path for imports
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from quant_strategy.base import Signal, Action, PortfolioState, Context
from quant_strategy.utils import calculate_max_drawdown


class RiskManager:
    """
    Global risk gates for the trading system.
    
    Validates signals against:
    - Position size limits
    - Sector/symbol concentration
    - Portfolio drawdown limits
    - Value at Risk (VaR) limits
    """
    
    def __init__(
        self,
        max_position_size: float = 0.20,
        max_sector_concentration: float = 0.40,
        max_drawdown: float = 0.15,
        var_confidence: float = 0.95,
        var_limit: float = 0.05
    ):
        """
        Initialize risk manager with limits.
        
        Args:
            max_position_size: Max % of portfolio in single position (default 20%)
            max_sector_concentration: Max % in single sector (default 40%)
            max_drawdown: Circuit breaker drawdown threshold (default 15%)
            var_confidence: VaR confidence level (default 95%)
            var_limit: Maximum VaR as % of portfolio (default 5%)
        """
        self.max_position_size = max_position_size
        self.max_sector_concentration = max_sector_concentration
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.var_limit = var_limit
        
        self.breach_log: List[Dict] = []
    
    def check_position_limits(
        self,
        portfolio: PortfolioState,
        signal: Signal,
        symbol: str,
        current_price: float,
        proposed_quantity: int
    ) -> Tuple[bool, str]:
        """
        Validate if new position would violate size limits.
        
        Args:
            portfolio: Current portfolio state
            signal: Trading signal
            symbol: Symbol to trade
            current_price: Current market price
            proposed_quantity: Number of shares to buy/sell
            
        Returns:
            (is_valid, reason) - True if allowed, False if breach
        """
        if portfolio.total_value == 0:
            return True, "Portfolio empty, first position allowed"
        
        # Calculate new position value
        current_position = portfolio.get_position(symbol)
        new_position = current_position + proposed_quantity
        new_position_value = abs(new_position * current_price)
        
        # Check position size limit
        position_pct = new_position_value / portfolio.total_value
        
        if position_pct > self.max_position_size:
            reason = (f"Position size breach: {position_pct:.1%} exceeds limit "
                     f"of {self.max_position_size:.1%}")
            self._log_breach("POSITION_SIZE", symbol, reason)
            return False, reason
        
        return True, "Position size within limits"
    
    def check_sector_concentration(
        self,
        portfolio: PortfolioState,
        symbol: str,
        sector: str,
        sector_map: Dict[str, str],
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if adding to this sector would violate concentration limits.
        
        Args:
            portfolio: Current portfolio
            symbol: New symbol
            sector: Sector of new symbol
            sector_map: {symbol: sector} mapping for all holdings
            current_prices: {symbol: price} for all holdings
            
        Returns:
            (is_valid, reason)
        """
        if portfolio.total_value == 0:
            return True, "First position"
        
        # Calculate current sector exposure
        sector_value = 0.0
        for sym, qty in portfolio.positions.items():
            if sector_map.get(sym) == sector:
                sector_value += abs(qty * current_prices.get(sym, 0.0))
        
        sector_pct = sector_value / portfolio.total_value
        
        if sector_pct > self.max_sector_concentration:
            reason = (f"Sector concentration breach: {sector} at {sector_pct:.1%} "
                     f"exceeds limit of {self.max_sector_concentration:.1%}")
            self._log_breach("SECTOR_CONCENTRATION", sector, reason)
            return False, reason
        
        return True, "Sector concentration within limits"
    
    def check_drawdown(self, portfolio: PortfolioState) -> Tuple[bool, str]:
        """
        Circuit breaker: Stop trading if drawdown exceeds threshold.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            (is_valid, reason) - False if circuit breaker triggered
        """
        current_dd = portfolio.current_drawdown
        
        if current_dd > self.max_drawdown:
            reason = (f"CIRCUIT BREAKER: Drawdown {current_dd:.1%} exceeds "
                     f"limit of {self.max_drawdown:.1%}")
            self._log_breach("DRAWDOWN", "PORTFOLIO", reason)
            return False, reason
        
        return True, f"Drawdown at {current_dd:.1%}, within limits"
    
    def calculate_var(
        self,
        positions: Dict[str, int],
        returns_history: pd.DataFrame,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk using historical simulation.
        
        Args:
            positions: {symbol: quantity}
            returns_history: DataFrame with columns=symbols, rows=daily returns
            confidence: VaR confidence level (uses self.var_confidence if None)
            
        Returns:
            VaR (maximum expected loss at confidence level)
        """
        if confidence is None:
            confidence = self.var_confidence
        
        # Filter only symbols we have positions in
        held_symbols = [s for s in positions.keys() if positions[s] != 0]
        if not held_symbols:
            return 0.0
        
        # Get returns for held symbols
        try:
            relevant_returns = returns_history[held_symbols]
        except KeyError:
            # Some symbols not in history
            available = [s for s in held_symbols if s in returns_history.columns]
            if not available:
                return 0.0
            relevant_returns = returns_history[available]
        
        # Create position weights vector
        weights = np.array([positions.get(s, 0) for s in relevant_returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (relevant_returns * weights).sum(axis=1)
        
        # VaR at confidence level (e.g., 95th percentile of losses)
        var_percentile = (1 - confidence) * 100
        var = np.percentile(portfolio_returns, var_percentile)
        
        return abs(var)
    
    def validate_signal(
        self,
        context: Context,
        signal: Signal,
        proposed_quantity: int = 100,
        sector: Optional[str] = None,
        sector_map: Optional[Dict[str, str]] = None,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive risk check for a signal.
        
        Args:
            context: Market context
            signal: Generated signal
            proposed_quantity: Shares to trade
            sector: Sector of symbol (optional)
            sector_map: Sector mappings (optional)
            current_prices: Price dictionary (optional)
            
        Returns:
            (is_valid, [reasons]) - True if all checks pass
        """
        if signal.action == Action.HOLD:
            return True, ["HOLD signal, no risk checks needed"]
        
        portfolio = context.portfolio
        if portfolio is None:
            return True, ["No portfolio provided, skipping risk checks"]
        
        reasons = []
        
        # 1. Drawdown check
        dd_ok, dd_reason = self.check_drawdown(portfolio)
        if not dd_ok:
            return False, [dd_reason]
        reasons.append(dd_reason)
        
        # 2. Position size check
        pos_ok, pos_reason = self.check_position_limits(
            portfolio, signal, context.symbol, context.price, proposed_quantity
        )
        if not pos_ok:
            return False, reasons + [pos_reason]
        reasons.append(pos_reason)
        
        # 3. Sector concentration (if provided)
        if sector and sector_map and current_prices:
            sector_ok, sector_reason = self.check_sector_concentration(
                portfolio, context.symbol, sector, sector_map, current_prices
            )
            if not sector_ok:
                return False, reasons + [sector_reason]
            reasons.append(sector_reason)
        
        return True, reasons
    
    def _log_breach(self, breach_type: str, entity: str, reason: str) -> None:
        """Log risk limit breach for auditing."""
        from datetime import datetime
        self.breach_log.append({
            'timestamp': datetime.now(),
            'type': breach_type,
            'entity': entity,
            'reason': reason
        })
    
    def get_breach_summary(self) -> pd.DataFrame:
        """Return DataFrame of all risk breaches for analysis."""
        if not self.breach_log:
            return pd.DataFrame()
        return pd.DataFrame(self.breach_log)
