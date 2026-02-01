"""
Backtesting & Risk Management Framework - CONSOLIDATED VERSION

A comprehensive backtesting system for AI-driven trading strategies with:
- Realistic execution modeling (slippage, costs, t+1 delays)
- Multi-layer risk management
- Professional-grade performance metrics
- Walk-forward validation
- Complete audit trails

Quick Start:
    from Backtesting_risk import BacktestEngine
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run(market_data, features, decisions)

File Structure (6 files total):
    - models.py: Data structures
    - backtesting.py: All engines (Execution + Risk + Portfolio + Backtest)
    - analysis.py: Performance metrics + Walk-forward validation
    - examples.py: Usage examples
    - __init__.py: This file
    - README_CONSOLIDATED.md: Complete documentation
"""

# Data models
from .models import (
    Action,
    TradeDecision,
    Position,
    ExecutedTrade,
    ClosedPosition,
    PortfolioState,
    AuditLogEntry,
    RiskMetrics,
)

# Backtesting engines (consolidated)
from .backtesting import (
    BacktestEngine,
    ExecutionEngine,
    ExecutionConfig,
    RiskEngine,
    RiskConfig,
    Portfolio,
)

# Analysis tools (consolidated)
from .analysis import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    WalkForwardValidator,
    WalkForwardResults,
    WindowResult,
)

__version__ = "2.0.0"  # Consolidated version

__all__ = [
    # Models
    "Action",
    "TradeDecision",
    "Position",
    "ExecutedTrade",
    "ClosedPosition",
    "PortfolioState",
    "AuditLogEntry",
    "RiskMetrics",
    # Core engines (from backtesting.py)
    "BacktestEngine",
    "ExecutionEngine",
    "ExecutionConfig",
    "Portfolio",
    "RiskEngine",
    "RiskConfig",
    # Analysis (from analysis.py)
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "WalkForwardValidator",
    "WalkForwardResults",
    "WindowResult",
]

