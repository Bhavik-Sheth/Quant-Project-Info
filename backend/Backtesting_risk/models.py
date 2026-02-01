"""
Data models for backtesting framework.

Defines core structures for trade decisions, positions, and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
import pandas as pd


class Action(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeDecision:
    """
    Represents a trading decision from upstream logic (AI agents/strategies).
    
    This is an advisory signal, not guaranteed execution.
    """
    timestamp: pd.Timestamp
    symbol: str
    action: Action
    position_size: float  # Fraction of portfolio capital (0.0 to 1.0)
    stop_loss: Optional[float] = None  # Percentage loss trigger (e.g., 0.05 for -5%)
    take_profit: Optional[float] = None  # Percentage gain trigger (e.g., 0.10 for +10%)
    max_holding_period: Optional[int] = None  # Maximum bars to hold
    
    def __post_init__(self):
        """Validate trade decision parameters."""
        if not isinstance(self.action, Action):
            self.action = Action(self.action)
        
        if not 0 <= self.position_size <= 1:
            raise ValueError(f"Position size must be between 0 and 1, got {self.position_size}")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"Stop loss must be positive, got {self.stop_loss}")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError(f"Take profit must be positive, got {self.take_profit}")
        
        if self.max_holding_period is not None and self.max_holding_period <= 0:
            raise ValueError(f"Max holding period must be positive, got {self.max_holding_period}")


@dataclass
class Position:
    """
    Represents an open trading position.
    
    Tracks entry details, unrealized PnL, and exit conditions.
    """
    symbol: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_holding_period: Optional[int] = None
    
    # Tracking fields
    current_price: float = 0.0
    holding_duration: int = 0
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost of the position."""
        return abs(self.quantity) * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.quantity > 0:  # Long position
            return (self.current_price - self.entry_price) * self.quantity
        else:  # Short position
            return (self.entry_price - self.current_price) * abs(self.quantity)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis
    
    def update(self, current_price: float, bars_elapsed: int = 1):
        """Update position with current market price."""
        self.current_price = current_price
        self.holding_duration += bars_elapsed
    
    def should_exit_stop_loss(self) -> bool:
        """Check if stop loss condition is met."""
        if self.stop_loss is None:
            return False
        return self.unrealized_pnl_pct <= -self.stop_loss
    
    def should_exit_take_profit(self) -> bool:
        """Check if take profit condition is met."""
        if self.take_profit is None:
            return False
        return self.unrealized_pnl_pct >= self.take_profit
    
    def should_exit_time(self) -> bool:
        """Check if maximum holding period is exceeded."""
        if self.max_holding_period is None:
            return False
        return self.holding_duration >= self.max_holding_period
    
    def should_exit(self) -> tuple[bool, str]:
        """
        Check all exit conditions.
        
        Returns:
            (should_exit, reason)
        """
        if self.should_exit_stop_loss():
            return True, "stop_loss"
        if self.should_exit_take_profit():
            return True, "take_profit"
        if self.should_exit_time():
            return True, "time_exit"
        return False, ""


@dataclass
class ExecutedTrade:
    """Record of an executed trade."""
    timestamp: pd.Timestamp
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    price: float
    slippage: float
    commission: float
    fees: float
    total_cost: float
    
    @property
    def gross_value(self) -> float:
        """Gross trade value before costs."""
        return abs(self.quantity) * self.price
    
    @property
    def net_value(self) -> float:
        """Net trade value after costs."""
        if self.action == "BUY":
            return -(self.gross_value + self.total_cost)
        else:
            return self.gross_value - self.total_cost


@dataclass
class ClosedPosition:
    """Record of a closed position with realized PnL."""
    symbol: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    holding_duration: int
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str  # "stop_loss", "take_profit", "time_exit", "manual"
    
    @property
    def is_win(self) -> bool:
        """Whether the trade was profitable."""
        return self.realized_pnl > 0
    
    @property
    def total_pnl(self) -> float:
        """Alias for realized_pnl (for compatibility)."""
        return self.realized_pnl
    
    @property
    def return_pct(self) -> float:
        """Alias for realized_pnl_pct (for compatibility)."""
        return self.realized_pnl_pct


@dataclass
class PortfolioState:
    """
    Current state of the portfolio.
    
    Tracks cash, positions, equity, and drawdown metrics.
    """
    timestamp: pd.Timestamp
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[ClosedPosition] = field(default_factory=list)
    peak_equity: float = 0.0
    
    def __post_init__(self):
        """Initialize peak equity if not set."""
        if self.peak_equity == 0.0:
            self.peak_equity = self.cash
    
    @property
    def positions_value(self) -> float:
        """Total market value of all open positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def equity(self) -> float:
        """Total portfolio equity (cash + positions)."""
        return self.cash + self.positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity
    
    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as percentage."""
        return self.drawdown * 100
    
    def update_peak(self):
        """Update peak equity if current equity is higher."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def add_position(self, position: Position):
        """Add a new position."""
        self.positions[position.symbol] = position
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove and return a position."""
        return self.positions.pop(symbol, None)
    
    def has_position(self, symbol: str) -> bool:
        """Check if holding a position in symbol."""
        return symbol in self.positions


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio or position."""
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk at 95% confidence
    max_drawdown: float = 0.0
    downside_deviation: float = 0.0
    beta: float = 0.0
    correlation_with_market: float = 0.0


@dataclass
class AuditLogEntry:
    """
    Single timestep audit log entry.
    
    Captures complete state for post-mortem analysis.
    """
    timestamp: pd.Timestamp
    
    # Market state
    market_data: Dict[str, float]  # symbol -> price
    features: Dict[str, float]  # feature_name -> value
    
    # Trade decisions received
    decisions_received: List[TradeDecision]
    
    # Risk checks
    risk_checks: Dict[str, bool]  # check_name -> passed
    rejected_decisions: List[tuple[TradeDecision, str]]  # (decision, rejection_reason)
    
    # Execution outcomes
    executed_trades: List[ExecutedTrade]
    closed_positions: List[ClosedPosition]
    
    # Portfolio state
    portfolio_equity: float
    portfolio_cash: float
    portfolio_positions_value: float
    portfolio_drawdown: float
    
    # Risk metrics
    portfolio_volatility: float
    portfolio_var: float
    
    # Model predictions and explainability (NEW for 10.4 compliance)
    model_predictions: Dict[str, Any] = field(default_factory=dict)  # What each model predicted
    agent_recommendations: Dict[str, Any] = field(default_factory=dict)  # What each agent recommended
    explainability_metrics: Dict[str, Any] = field(default_factory=dict)  # SHAP values, feature importance
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'market_data': self.market_data,
            'features': self.features,
            'decisions_count': len(self.decisions_received),
            'risk_checks': self.risk_checks,
            'rejections_count': len(self.rejected_decisions),
            'executions_count': len(self.executed_trades),
            'closes_count': len(self.closed_positions),
            'equity': self.portfolio_equity,
            'cash': self.portfolio_cash,
            'positions_value': self.portfolio_positions_value,
            'drawdown': self.portfolio_drawdown,
            'volatility': self.portfolio_volatility,
            'var': self.portfolio_var,
            'model_predictions': self.model_predictions,
            'agent_recommendations': self.agent_recommendations,
            'explainability_metrics': self.explainability_metrics,
        }
