"""
Quantitative Strategy Library

AI-Augmented trading strategy framework with:
- Dynamic LLM-powered orchestration
- White-box explainability
- Strict temporal integrity
- MongoDB integration
"""

__version__ = "1.0.0"

# Core abstractions
from quant_strategy.base import (
    Context,
    Signal,
    Action,
    Regime,
    StrategyType,
    BaseStrategy,
    PortfolioState
)

# Utilities
from quant_strategy.utils import (
    black_scholes,
    kelly_criterion,
    volatility_position_sizer,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

# Components
from quant_strategy.components.risk_manager import RiskManager
from quant_strategy.components.ensemble import StrategyOrchestrator, ConflictResolver

# Strategies
from quant_strategy.strategies.technical import (
    RSIStrategy,
    MovingAverageCrossStrategy,
    BollingerBandsStrategy
)
from quant_strategy.strategies.ml_enhanced import MLSignalFilter, MLDirectStrategy
from quant_strategy.strategies.options import VolArbitrageStrategy

# Engine
from quant_strategy.engine import BacktestEngine

__all__ = [
    # Core
    'Context',
    'Signal',
    'Action',
    'Regime',
    'StrategyType',
    'BaseStrategy',
    'PortfolioState',
    
    # Utils
    'black_scholes',
    'kelly_criterion',
    'volatility_position_sizer',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    
    # Components
    'RiskManager',
    'StrategyOrchestrator',
    'ConflictResolver',
    
    # Strategies
    'RSIStrategy',
    'MovingAverageCrossStrategy',
    'BollingerBandsStrategy',
    'MLSignalFilter',
    'MLDirectStrategy',
    'VolArbitrageStrategy',
    
    # Engine
    'BacktestEngine'
]
