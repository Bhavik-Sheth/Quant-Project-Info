"""
Base Module for Quantitative Strategy Library

This module provides the foundational abstractions for the strategy system:
- Enums for actions, regimes, and strategy types
- Dataclasses for context and signals
- Abstract base class for all strategies

Temporal Integrity: Context represents state at time t, Signal is decision at time t
Execution happens at t+1 (handled by BacktestEngine)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd


# ==================== ENUMS ====================

class Action(Enum):
    """Trading actions that can be signaled"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    HEDGE = "HEDGE"
    CLOSE = "CLOSE"  # Close existing position


class Regime(Enum):
    """Market regime classifications"""
    BULLISH = "BULLISH"           # Strong uptrend
    BEARISH = "BEARISH"           # Strong downtrend
    RANGING = "RANGING"           # Sideways/mean-reverting
    VOLATILE = "VOLATILE"         # High volatility, uncertain direction
    CRISIS = "CRISIS"             # Extreme market stress


class StrategyType(Enum):
    """Strategy category identifiers"""
    TECHNICAL = "TECHNICAL"       # Pure technical analysis
    ML_ENHANCED = "ML_ENHANCED"   # ML-filtered strategies
    OPTIONS = "OPTIONS"           # Volatility arbitrage
    HYBRID = "HYBRID"             # Multiple approaches


# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class Context:
    """
    Immutable snapshot of market state at time t.
    
    CRITICAL: This object must ONLY contain information available at time t.
    No look-ahead bias allowed - execution happens at t+1.
    
    Attributes:
        timestamp: Current time
        symbol: Trading symbol (e.g., "AAPL", "NIFTY")
        price: Current price (close or mid-price)
        features: Technical indicators as dict (e.g., {'RSI': 65.2, 'SMA_50': 150.3})
        ml_predictions: ML model outputs (e.g., {'direction_confidence': 0.85})
        portfolio: Current portfolio state
        option_chain: Available options data (optional)
        current_regime: Detected market regime
    """
    timestamp: datetime
    symbol: str
    price: float
    features: Dict[str, float] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    portfolio: Optional[Any] = None  # Portfolio object from engine
    option_chain: Optional[Dict[str, Any]] = None
    current_regime: Regime = Regime.RANGING
    
    def __post_init__(self):
        """Validation to prevent look-ahead bias"""
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be datetime object")


@dataclass
class Signal:
    """
    Output from a strategy with full explainability.
    
    Every signal must include a natural language reason for "White Box" transparency.
    
    Attributes:
        action: What to do (BUY/SELL/HOLD/HEDGE)
        confidence: Strategy's confidence (0.0 to 1.0)
        reason: Natural language explanation (REQUIRED)
        metadata: Additional structured data
        strategy_name: Which strategy generated this
        timestamp: When signal was generated
    """
    action: Action
    confidence: float
    reason: str
    strategy_name: str = "UnknownStrategy"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if not self.reason or not self.reason.strip():
            raise ValueError("Signal MUST have a non-empty reason for explainability")
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PortfolioState:
    """
    Current portfolio holdings and statistics.
    
    Used by strategies to consider portfolio constraints and risk.
    """
    cash: float
    positions: Dict[str, int]  # {symbol: quantity}
    equity_curve: List[float] = field(default_factory=list)
    total_value: float = 0.0
    current_drawdown: float = 0.0
    
    def get_position(self, symbol: str) -> int:
        """Get position size for a symbol"""
        return self.positions.get(symbol, 0)
    
    def update_value(self, current_prices: Dict[str, float]) -> None:
        """Recalculate total portfolio value"""
        position_value = sum(
            qty * current_prices.get(sym, 0.0)
            for sym, qty in self.positions.items()
        )
        self.total_value = self.cash + position_value
        self.equity_curve.append(self.total_value)
        
        # Update drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            self.current_drawdown = (peak - self.total_value) / peak if peak > 0 else 0.0


# ==================== ABSTRACT BASE CLASS ====================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All concrete strategies MUST inherit from this class and implement:
    - generate_signal(context): Core signal generation logic
    
    Optional to override:
    - adjust_parameters(volatility): Dynamic parameter tuning
    """
    
    def __init__(self, name: str, strategy_type: StrategyType):
        """
        Initialize strategy.
        
        Args:
            name: Unique identifier for this strategy instance
            strategy_type: Category of strategy
        """
        self.name = name
        self.strategy_type = strategy_type
        self.parameters: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def generate_signal(self, context: Context) -> Signal:
        """
        Generate trading signal based on current context.
        
        MUST be implemented by all child strategies.
        
        Args:
            context: Market state at time t
            
        Returns:
            Signal with action, confidence, and natural language reason
            
        Raises:
            NotImplementedError: If not overridden by child class
        """
        raise NotImplementedError("Strategies must implement generate_signal()")
    
    def adjust_parameters(self, volatility: float) -> None:
        """
        Dynamically adjust strategy parameters based on market volatility.
        
        Default implementation: Scale stop-loss and position size by volatility.
        Override in child classes for custom behavior.
        
        Args:
            volatility: Current market volatility (e.g., 0.02 for 2%)
        """
        # Example: Widen stop-loss in high volatility
        if 'stop_loss_pct' in self.parameters:
            base_stop = self.parameters.get('base_stop_loss', 0.02)
            self.parameters['stop_loss_pct'] = base_stop * (1 + volatility * 10)
        
        # Example: Reduce position size in high volatility
        if 'position_size' in self.parameters:
            base_size = self.parameters.get('base_position_size', 1.0)
            vol_factor = max(0.5, 1.0 - volatility * 5)  # Reduce up to 50%
            self.parameters['position_size'] = base_size * vol_factor
    
    def log_performance(self, context: Context, signal: Signal, outcome: Optional[float] = None) -> None:
        """
        Log signal and outcome for strategy performance tracking.
        
        Args:
            context: Input context
            signal: Generated signal
            outcome: Actual P&L or accuracy (optional, added later)
        """
        self.performance_history.append({
            'timestamp': context.timestamp,
            'symbol': context.symbol,
            'signal': signal.action.value,
            'confidence': signal.confidence,
            'outcome': outcome
        })
    
    def get_performance_metrics(self, window: int = 50) -> Dict[str, float]:
        """
        Calculate recent performance statistics.
        
        Args:
            window: Number of recent signals to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        recent = self.performance_history[-window:]
        if not recent:
            return {'avg_outcome': 0.0, 'hit_rate': 0.0, 'sample_size': 0}
        
        outcomes = [r['outcome'] for r in recent if r['outcome'] is not None]
        if not outcomes:
            return {'avg_outcome': 0.0, 'hit_rate': 0.0, 'sample_size': 0}
        
        return {
            'avg_outcome': sum(outcomes) / len(outcomes),
            'hit_rate': sum(1 for o in outcomes if o > 0) / len(outcomes),
            'sample_size': len(outcomes)
        }
    
    def __repr__(self) -> str:
        return f"{self.name} ({self.strategy_type.value})"
