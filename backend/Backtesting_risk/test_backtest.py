"""
Unit tests for Backtesting & Risk Management framework.

Run with: python -m pytest Backtesting_risk/test_backtest.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Backtesting_risk import (
    BacktestEngine,
    TradeDecision,
    WalkForwardValidator
)
from Backtesting_risk.models import Action, Position, PortfolioState, ExecutedTrade
from Backtesting_risk.execution import ExecutionEngine, ExecutionConfig
from Backtesting_risk.risk import RiskEngine, RiskConfig
from Backtesting_risk.portfolio import Portfolio
from Backtesting_risk.performance import PerformanceAnalyzer


# Test Fixtures

@pytest.fixture
def sample_market_data():
    """Generate sample market data."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 100.0 + np.random.randn(),
                'volume': 1000000
            })
    
    df = pd.DataFrame(data)
    return df.set_index(['timestamp', 'symbol'])


@pytest.fixture
def sample_features(sample_market_data):
    """Generate sample features."""
    features = []
    for symbol in sample_market_data.index.get_level_values(1).unique():
        symbol_data = sample_market_data.xs(symbol, level=1)
        for timestamp in symbol_data.index:
            features.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'returns': 0.01,
                'volatility': 0.2
            })
    
    df = pd.DataFrame(features)
    return df.set_index(['timestamp', 'symbol'])


# Model Tests

class TestTradeDecision:
    """Test TradeDecision model."""
    
    def test_valid_decision(self):
        decision = TradeDecision(
            timestamp=pd.Timestamp('2020-01-01'),
            symbol='AAPL',
            action=Action.BUY,
            position_size=0.05
        )
        assert decision.symbol == 'AAPL'
        assert decision.action == Action.BUY
        assert decision.position_size == 0.05
    
    def test_invalid_position_size(self):
        with pytest.raises(ValueError):
            TradeDecision(
                timestamp=pd.Timestamp('2020-01-01'),
                symbol='AAPL',
                action=Action.BUY,
                position_size=1.5  # Invalid: > 1
            )
    
    def test_action_conversion(self):
        decision = TradeDecision(
            timestamp=pd.Timestamp('2020-01-01'),
            symbol='AAPL',
            action='BUY',  # String instead of enum
            position_size=0.05
        )
        assert decision.action == Action.BUY


class TestPosition:
    """Test Position model."""
    
    def test_position_creation(self):
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=10,
            stop_loss=0.05,
            take_profit=0.10
        )
        assert position.symbol == 'AAPL'
        assert position.quantity == 10
    
    def test_unrealized_pnl_long(self):
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=10,
            current_price=110.0
        )
        assert position.unrealized_pnl == 100.0  # (110-100) * 10
        assert position.unrealized_pnl_pct == 0.1  # 10%
    
    def test_stop_loss_trigger(self):
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=10,
            stop_loss=0.05,
            current_price=94.0
        )
        assert position.should_exit_stop_loss()
    
    def test_take_profit_trigger(self):
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=10,
            take_profit=0.10,
            current_price=111.0
        )
        assert position.should_exit_take_profit()
    
    def test_time_exit_trigger(self):
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=10,
            max_holding_period=5,
            current_price=100.0,
            holding_duration=5
        )
        assert position.should_exit_time()


# Execution Tests

class TestExecutionEngine:
    """Test ExecutionEngine."""
    
    def test_slippage_calculation_buy(self):
        engine = ExecutionEngine(ExecutionConfig(slippage_pct=0.001))
        price = engine.apply_slippage(100.0, Action.BUY)
        assert price == 100.1  # 100 * (1 + 0.001)
    
    def test_slippage_calculation_sell(self):
        engine = ExecutionEngine(ExecutionConfig(slippage_pct=0.001))
        price = engine.apply_slippage(100.0, Action.SELL)
        assert price == 99.9  # 100 * (1 - 0.001)
    
    def test_trade_cost_calculation(self):
        config = ExecutionConfig(
            commission_pct=0.001,
            exchange_fee_pct=0.0005,
            min_commission=1.0
        )
        engine = ExecutionEngine(config)
        
        commission, fees, total = engine.calculate_trade_cost(10, 100.0)
        assert commission == 1.0  # max(1000 * 0.001, 1.0)
        assert fees == 0.5  # 1000 * 0.0005
        assert total == 1.5


# Portfolio Tests

class TestPortfolio:
    """Test Portfolio."""
    
    def test_initialization(self):
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.cash == 100000
        assert portfolio.equity == 100000
        assert len(portfolio.positions) == 0
    
    def test_invalid_initial_capital(self):
        with pytest.raises(ValueError):
            Portfolio(initial_capital=-1000)
    
    def test_equity_calculation(self):
        portfolio = Portfolio(initial_capital=100000)
        
        # Add a position
        position = Position(
            symbol='AAPL',
            entry_timestamp=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            quantity=100,
            current_price=110.0
        )
        portfolio.positions['AAPL'] = position
        
        # Reduce cash by position cost
        portfolio.cash -= 10000
        
        assert portfolio.positions_value == 11000  # 100 * 110
        assert portfolio.equity == 101000  # 90000 + 11000
    
    def test_drawdown_calculation(self):
        portfolio = Portfolio(initial_capital=100000)
        portfolio.peak_equity = 120000
        portfolio.cash = 90000  # Lost 30k from peak
        
        assert portfolio.drawdown == -0.25  # -30k / 120k


# Risk Tests

class TestRiskEngine:
    """Test RiskEngine."""
    
    def test_position_size_check(self):
        risk_engine = RiskEngine(RiskConfig(max_position_size_pct=0.10))
        
        decision = TradeDecision(
            timestamp=pd.Timestamp('2020-01-01'),
            symbol='AAPL',
            action=Action.BUY,
            position_size=0.15  # Exceeds 10% limit
        )
        
        portfolio_state = PortfolioState(
            timestamp=pd.Timestamp('2020-01-01'),
            cash=100000,
            peak_equity=100000
        )
        
        approved, reasons = risk_engine.check_trade_decision(decision, portfolio_state)
        assert not approved
        assert len(reasons) > 0
    
    def test_drawdown_circuit_breaker(self):
        risk_engine = RiskEngine(RiskConfig(
            max_drawdown_pct=0.20,
            circuit_breaker_enabled=True
        ))
        
        portfolio_state = PortfolioState(
            timestamp=pd.Timestamp('2020-01-01'),
            cash=80000,
            peak_equity=100000
        )
        
        warning, breaker, dd = risk_engine.check_drawdown(portfolio_state)
        assert not warning  # 20% DD is at limit, not warning level
        assert breaker
        assert risk_engine.circuit_breaker_triggered


# Performance Tests

class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer."""
    
    def test_total_return_calculation(self):
        analyzer = PerformanceAnalyzer()
        
        equity_curve = pd.DataFrame({
            'equity': [100000, 110000, 115000]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        total_return = analyzer._calculate_total_return(equity_curve, 100000)
        assert total_return == 0.15  # 15% return
    
    def test_max_drawdown_calculation(self):
        analyzer = PerformanceAnalyzer()
        
        equity_curve = pd.DataFrame({
            'equity': [100000, 120000, 90000, 110000]
        }, index=pd.date_range('2020-01-01', periods=4))
        
        max_dd = analyzer._calculate_max_drawdown(equity_curve)
        assert max_dd == 0.25  # -30k from peak of 120k
    
    def test_sharpe_ratio_calculation(self):
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        sharpe = analyzer._calculate_sharpe_ratio(returns, 0.02)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)


# Integration Tests

class TestBacktestEngine:
    """Test BacktestEngine integration."""
    
    def test_basic_backtest(self, sample_market_data, sample_features):
        decisions = [
            TradeDecision(
                timestamp=pd.Timestamp('2020-01-05'),
                symbol='AAPL',
                action=Action.BUY,
                position_size=0.05,
                stop_loss=0.05,
                take_profit=0.10
            )
        ]
        
        engine = BacktestEngine(initial_capital=100000, enable_logging=False)
        metrics = engine.run(
            market_data=sample_market_data,
            features=sample_features,
            trade_decisions=decisions
        )
        
        assert metrics is not None
        assert isinstance(metrics.total_return, float)
        assert engine.portfolio.equity > 0
    
    def test_multiple_trades(self, sample_market_data, sample_features):
        decisions = []
        dates = sample_market_data.index.get_level_values(0).unique()[10:20]
        
        for date in dates:
            decisions.append(TradeDecision(
                timestamp=date,
                symbol='AAPL',
                action=Action.BUY,
                position_size=0.03,
                stop_loss=0.05,
                take_profit=0.10,
                max_holding_period=5
            ))
        
        engine = BacktestEngine(initial_capital=100000, enable_logging=False)
        metrics = engine.run(
            market_data=sample_market_data,
            features=sample_features,
            trade_decisions=decisions
        )
        
        assert engine.portfolio.get_total_trades_count() > 0


class TestWalkForwardValidator:
    """Test WalkForwardValidator."""
    
    def test_window_generation_rolling(self):
        timestamps = pd.date_range('2020-01-01', periods=100, freq='D')
        
        validator = WalkForwardValidator(
            train_window_size=30,
            test_window_size=10,
            step_size=10
        )
        
        windows = validator.generate_windows(timestamps, expanding=False)
        
        assert len(windows) > 0
        for train, test in windows:
            assert len(train) == 30
            assert len(test) <= 10
            assert train[-1] < test[0]  # No overlap
    
    def test_window_generation_expanding(self):
        timestamps = pd.date_range('2020-01-01', periods=100, freq='D')
        
        validator = WalkForwardValidator(
            train_window_size=30,
            test_window_size=10,
            step_size=10,
            min_train_size=30
        )
        
        windows = validator.generate_windows(timestamps, expanding=True)
        
        assert len(windows) > 0
        prev_train_size = 0
        for train, test in windows:
            assert len(train) >= 30  # At least min_train_size
            assert len(train) >= prev_train_size  # Expanding
            prev_train_size = len(train)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
