"""
Backtesting & Risk Management Framework - Core Engine

Consolidates:
- Execution simulation (slippage, costs)
- Portfolio management
- Risk management
- Backtest orchestration
"""

from typing import List, Dict, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

from .models import (
    TradeDecision, Action, Position, ExecutedTrade, 
    ClosedPosition, AuditLogEntry, PortfolioState
)


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

@dataclass
class ExecutionConfig:
    """Execution simulation configuration."""
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_pct: float = 0.001  # 0.1% commission
    exchange_fee_pct: float = 0.0001  # 0.01% exchange fee


class ExecutionEngine:
    """Simulates realistic trade execution with slippage and costs."""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.pending_orders: List[Tuple[TradeDecision, pd.Timestamp, float]] = []
    
    def schedule_order(self, decision: TradeDecision, timestamp: pd.Timestamp, portfolio_equity: float):
        """Schedule order for t+1 execution."""
        self.pending_orders.append((decision, timestamp, portfolio_equity))
    
    def execute_pending_orders(self, timestamp: pd.Timestamp, market_data: pd.DataFrame) -> List[ExecutedTrade]:
        """Execute all pending orders at current market prices."""
        executed = []
        remaining = []
        
        for decision, order_time, equity in self.pending_orders:
            if decision.symbol in market_data.index:
                current_price = market_data.loc[decision.symbol, 'close']
                if not pd.isna(current_price):
                    trade = self.execute_market_order(
                        decision.symbol, decision.action, 
                        decision.position_size * equity / float(current_price),
                        float(current_price), timestamp
                    )
                    executed.append(trade)
                else:
                    remaining.append((decision, order_time, equity))
            else:
                remaining.append((decision, order_time, equity))
        
        self.pending_orders = remaining
        return executed
    
    def execute_market_order(self, symbol: str, action: Action, quantity: float,
                            current_price: float, timestamp: pd.Timestamp) -> ExecutedTrade:
        """Execute market order with slippage and costs."""
        # Apply slippage
        if action == Action.BUY:
            execution_price = current_price * (1 + self.config.slippage_pct)
        else:
            execution_price = current_price * (1 - self.config.slippage_pct)
        
        # Calculate costs
        trade_value = abs(quantity) * execution_price
        commission = trade_value * self.config.commission_pct
        exchange_fee = trade_value * self.config.exchange_fee_pct
        total_cost = commission + exchange_fee
        
        # Net value (negative for buys, positive for sells)
        if action == Action.BUY:
            net_value = -(trade_value + total_cost)
        else:
            net_value = trade_value - total_cost
        
        return ExecutedTrade(
            timestamp=timestamp,
            symbol=symbol,
            action=action.value,
            quantity=quantity if action == Action.BUY else -quantity,
            price=execution_price,
            slippage=abs(execution_price - current_price),
            commission=commission,
            exchange_fee=exchange_fee,
            total_cost=total_cost,
            net_value=net_value
        )
    
    def clear_pending_orders(self):
        """Clear all pending orders."""
        self.pending_orders.clear()


# ============================================================================
# RISK ENGINE
# ============================================================================

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_pct: float = 0.10  # 10% per position
    max_total_positions: int = 20
    max_sector_exposure_pct: float = 0.30  # 30% per sector
    max_portfolio_volatility: float = 0.20  # 20% annualized
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    volatility_lookback: int = 30  # Days for volatility calculation


class RiskEngine:
    """Multi-layer risk management system."""
    
    def __init__(self, config: Optional[RiskConfig] = None, sector_map: Optional[Dict[str, str]] = None):
        self.config = config or RiskConfig()
        self.sector_map = sector_map or {}
        self.rejections: List[Dict] = []
        self.circuit_breaker_triggered = False
        self.returns_history: List[float] = []
    
    def check_trade_decision(self, decision: TradeDecision, portfolio_state: PortfolioState,
                            market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate trade decision against all risk constraints."""
        reasons = []
        
        # Circuit breaker check
        if self.circuit_breaker_triggered:
            reasons.append("Circuit breaker triggered")
        
        # Position size check
        if decision.position_size > self.config.max_position_size_pct:
            reasons.append(f"Position size {decision.position_size:.1%} exceeds max {self.config.max_position_size_pct:.1%}")
        
        # Max positions check
        if len(portfolio_state.positions) >= self.config.max_total_positions:
            reasons.append(f"Max positions ({self.config.max_total_positions}) reached")
        
        # Sector exposure check
        if self.sector_map:
            sector = self.sector_map.get(decision.symbol)
            if sector:
                sector_exposure = self._calculate_sector_exposure(portfolio_state, sector)
                new_exposure = sector_exposure + decision.position_size
                if new_exposure > self.config.max_sector_exposure_pct:
                    reasons.append(f"Sector exposure {new_exposure:.1%} exceeds max {self.config.max_sector_exposure_pct:.1%}")
        
        # Cash check
        required_cash = portfolio_state.equity * decision.position_size
        if required_cash > portfolio_state.cash:
            reasons.append(f"Insufficient cash: need ${required_cash:,.0f}, have ${portfolio_state.cash:,.0f}")
        
        # Log rejection if failed
        if reasons:
            self.rejections.append({
                'timestamp': decision.timestamp,
                'symbol': decision.symbol,
                'action': decision.action.value,
                'reason': "; ".join(reasons)
            })
            return False, reasons
        
        return True, []
    
    def should_reduce_exposure(self, portfolio_state: PortfolioState, 
                               recent_returns: Optional[pd.Series] = None) -> Tuple[bool, str, float]:
        """Check if portfolio-level risk reduction is needed."""
        # Drawdown check (circuit breaker)
        if abs(portfolio_state.drawdown) >= self.config.max_drawdown_pct:
            self.circuit_breaker_triggered = True
            return True, f"Circuit breaker: drawdown {abs(portfolio_state.drawdown):.1%}", 0.0
        
        # Volatility check
        if recent_returns is not None and len(recent_returns) >= 10:
            volatility = recent_returns.std() * np.sqrt(252)
            if volatility > self.config.max_portfolio_volatility:
                return True, f"Volatility {volatility:.1%} exceeds {self.config.max_portfolio_volatility:.1%}", 0.5
        
        return False, "", 1.0
    
    def _calculate_sector_exposure(self, portfolio_state: PortfolioState, sector: str) -> float:
        """Calculate current exposure to a sector."""
        sector_value = sum(
            pos['value'] for symbol, pos in portfolio_state.positions.items()
            if self.sector_map.get(symbol) == sector
        )
        return sector_value / portfolio_state.equity if portfolio_state.equity > 0 else 0.0
    
    def update_returns_history(self, period_return: float):
        """Update returns history for volatility calculation."""
        self.returns_history.append(period_return)
        if len(self.returns_history) > self.config.volatility_lookback * 2:
            self.returns_history = self.returns_history[-self.config.volatility_lookback * 2:]
    
    def get_risk_metrics(self, portfolio_state: PortfolioState, 
                        recent_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate current risk metrics."""
        metrics = {
            'drawdown': abs(portfolio_state.drawdown),
            'positions_count': len(portfolio_state.positions),
            'volatility': 0.0,
            'var_95': 0.0,
        }
        
        if recent_returns is not None and len(recent_returns) >= 10:
            metrics['volatility'] = recent_returns.std() * np.sqrt(252)
            metrics['var_95'] = abs(recent_returns.quantile(0.05))
        
        return metrics
    
    def get_rejection_summary(self) -> pd.DataFrame:
        """Get summary of rejected trades."""
        if not self.rejections:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'reason'])
        return pd.DataFrame(self.rejections)
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self.circuit_breaker_triggered = False


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class Portfolio:
    """Portfolio state management and tracking."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[ClosedPosition] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        return self.cash + self.positions_value
    
    @property
    def positions_value(self) -> float:
        """Current value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity
    
    def update_positions(self, market_data: pd.DataFrame):
        """Update all positions with current market prices."""
        for symbol, position in self.positions.items():
            if symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                if not pd.isna(current_price):
                    position.update(float(current_price))
    
    def execute_trade(self, trade: ExecutedTrade):
        """Execute trade and update cash."""
        self.cash += trade.net_value
        self.total_trades += 1
    
    def open_position(self, trade: ExecutedTrade, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None, max_holding_period: Optional[int] = None):
        """Open new position."""
        position = Position(
            symbol=trade.symbol, entry_timestamp=trade.timestamp,
            entry_price=trade.price, quantity=trade.quantity,
            stop_loss=stop_loss, take_profit=take_profit,
            max_holding_period=max_holding_period, current_price=trade.price
        )
        self.positions[trade.symbol] = position
    
    def close_position(self, symbol: str, exit_trade: ExecutedTrade, 
                      exit_reason: str = "manual") -> Optional[ClosedPosition]:
        """Close position and record PnL."""
        if symbol not in self.positions:
            return None
        
        position = self.positions.pop(symbol)
        
        # Calculate PnL
        if position.quantity > 0:
            realized_pnl = (exit_trade.price - position.entry_price) * abs(position.quantity)
        else:
            realized_pnl = (position.entry_price - exit_trade.price) * abs(position.quantity)
        
        realized_pnl -= exit_trade.total_cost
        realized_pnl_pct = realized_pnl / position.cost_basis if position.cost_basis > 0 else 0.0
        
        closed = ClosedPosition(
            symbol=symbol, entry_timestamp=position.entry_timestamp,
            exit_timestamp=exit_trade.timestamp, entry_price=position.entry_price,
            exit_price=exit_trade.price, quantity=position.quantity,
            holding_duration=position.holding_duration, realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct, exit_reason=exit_reason
        )
        
        self.closed_positions.append(closed)
        
        if closed.is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        return closed
    
    def check_exit_conditions(self, market_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """Check exit conditions for all positions."""
        to_exit = []
        for symbol, position in self.positions.items():
            should_exit, reason = position.should_exit()
            if should_exit:
                to_exit.append((symbol, reason))
        return to_exit
    
    def update_equity_curve(self, timestamp: pd.Timestamp):
        """Record equity in curve."""
        self.equity_curve.append((timestamp, self.equity))
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = abs(self.drawdown)
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=['equity'])
        timestamps, equities = zip(*self.equity_curve)
        return pd.DataFrame({'equity': equities}, 
                          index=pd.DatetimeIndex(timestamps, name='timestamp'))
    
    def get_drawdown_series(self) -> pd.Series:
        """Calculate drawdown series."""
        equity_df = self.get_equity_dataframe()
        if equity_df.empty:
            return pd.Series(dtype=float)
        peak = equity_df['equity'].cummax()
        return (equity_df['equity'] - peak) / peak
    
    def get_state(self, timestamp: pd.Timestamp) -> PortfolioState:
        """Get current portfolio state."""
        return PortfolioState(
            timestamp=timestamp, equity=self.equity, cash=self.cash,
            positions_value=self.positions_value,
            positions={s: {'value': p.market_value, 'pnl': p.unrealized_pnl} 
                      for s, p in self.positions.items()},
            drawdown=self.drawdown
        )
    
    def get_trade_log_dataframe(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if not self.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame([{
            'symbol': p.symbol, 'entry_timestamp': p.entry_timestamp,
            'exit_timestamp': p.exit_timestamp, 'entry_price': p.entry_price,
            'exit_price': p.exit_price, 'quantity': p.quantity,
            'holding_duration': p.holding_duration, 'realized_pnl': p.realized_pnl,
            'realized_pnl_pct': p.realized_pnl_pct, 'exit_reason': p.exit_reason,
            'is_win': p.is_win
        } for p in self.closed_positions])
    
    def reset(self):
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_positions.clear()
        self.equity_curve.clear()
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0


# ============================================================================
# BACKTEST ENGINE (Orchestrator)
# ============================================================================

class BacktestEngine:
    """Main backtesting orchestrator."""
    
    def __init__(self, initial_capital: float, execution_config: Optional[ExecutionConfig] = None,
                 risk_config: Optional[RiskConfig] = None, sector_map: Optional[Dict[str, str]] = None,
                 risk_free_rate: float = 0.02, enable_logging: bool = True):
        
        self.portfolio = Portfolio(initial_capital)
        self.execution_engine = ExecutionEngine(execution_config)
        self.risk_engine = RiskEngine(risk_config, sector_map)
        self.initial_capital = initial_capital
        self.enable_logging = enable_logging
        self.audit_log: List[AuditLogEntry] = []
        self.returns_series: List[float] = []
        self.model_predictions_callback: Optional[Callable] = None
        self.explainability_callback: Optional[Callable] = None
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            logging.basicConfig(level=logging.INFO, 
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def run(self, market_data: pd.DataFrame, features: pd.DataFrame,
            trade_decisions: List[TradeDecision],
            model_predictions_callback: Optional[Callable] = None,
            explainability_callback: Optional[Callable] = None) -> 'PerformanceMetrics':
        """Run backtest simulation."""
        from .analysis import PerformanceAnalyzer
        
        self.logger.info("Starting backtest simulation")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        timestamps = sorted(market_data.index.get_level_values(0).unique())
        self.logger.info(f"Simulating {len(timestamps)} time periods")
        
        decisions_by_time = {d.timestamp: [] for d in trade_decisions}
        for d in trade_decisions:
            decisions_by_time[d.timestamp].append(d)
        
        prev_equity = self.initial_capital
        
        for i, timestamp in enumerate(timestamps):
            market_slice = market_data.loc[timestamp] if isinstance(market_data.index, pd.MultiIndex) else market_data.loc[[timestamp]]
            
            self.portfolio.update_positions(market_slice)
            self._process_position_exits(timestamp, market_slice)
            
            if i > 0:
                executed_trades = self.execution_engine.execute_pending_orders(timestamp, market_slice)
                for trade in executed_trades:
                    self.portfolio.execute_trade(trade)
                    if trade.action == "BUY" and not trade.symbol in self.portfolio.positions:
                        self.portfolio.open_position(trade)
                        self.logger.info(f"Opened {trade.symbol}: {trade.quantity:.2f} @ ${trade.price:.2f}")
            
            decisions = decisions_by_time.get(timestamp, [])
            self._process_trade_decisions(decisions, timestamp, market_slice)
            
            should_reduce, reason, factor = self.risk_engine.should_reduce_exposure(
                self.portfolio.get_state(timestamp),
                pd.Series(self.returns_series[-30:]) if len(self.returns_series) >= 30 else None
            )
            
            if should_reduce:
                self.logger.warning(f"Risk reduction triggered: {reason}")
                self._reduce_exposure(factor, timestamp, market_slice)
            
            self.portfolio.update_equity_curve(timestamp)
            
            current_equity = self.portfolio.equity
            if prev_equity > 0:
                period_return = (current_equity - prev_equity) / prev_equity
                self.returns_series.append(period_return)
                self.risk_engine.update_returns_history(period_return)
            prev_equity = current_equity
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Progress: {i+1}/{len(timestamps)} | Equity: ${current_equity:,.2f} | Positions: {len(self.portfolio.positions)}")
        
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(
            self.portfolio.get_equity_dataframe(),
            self.portfolio.closed_positions,
            self.initial_capital
        )
        
        self.logger.info("Backtest complete")
        self.logger.info(f"Final equity: ${self.portfolio.equity:,.2f}")
        self.logger.info(f"Total return: {metrics.total_return:.2%}")
        self.logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        return metrics
    
    def _process_position_exits(self, timestamp: pd.Timestamp, market_data: pd.DataFrame):
        """Check and process position exits."""
        to_exit = self.portfolio.check_exit_conditions(market_data)
        for symbol, reason in to_exit:
            position = self.portfolio.positions.get(symbol)
            if position and symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                if not pd.isna(current_price):
                    exit_action = Action.SELL if position.quantity > 0 else Action.BUY
                    exit_trade = self.execution_engine.execute_market_order(
                        symbol, exit_action, abs(position.quantity), float(current_price), timestamp
                    )
                    self.portfolio.execute_trade(exit_trade)
                    closed = self.portfolio.close_position(symbol, exit_trade, reason)
                    if closed:
                        self.logger.info(f"Closed {symbol}: {reason} | PnL: ${closed.realized_pnl:,.2f} ({closed.realized_pnl_pct:.2%})")
    
    def _process_trade_decisions(self, decisions: List[TradeDecision], 
                                timestamp: pd.Timestamp, market_data: pd.DataFrame):
        """Process trade decisions through risk checks."""
        portfolio_state = self.portfolio.get_state(timestamp)
        for decision in decisions:
            if decision.action == Action.HOLD:
                continue
            passed, reasons = self.risk_engine.check_trade_decision(decision, portfolio_state, market_data)
            if passed:
                self.execution_engine.schedule_order(decision, timestamp, self.portfolio.equity)
    
    def _reduce_exposure(self, reduction_factor: float, timestamp: pd.Timestamp, market_data: pd.DataFrame):
        """Reduce portfolio exposure."""
        if reduction_factor == 0.0:
            symbols_to_close = list(self.portfolio.positions.keys())
        else:
            positions = list(self.portfolio.positions.values())
            positions.sort(key=lambda p: p.unrealized_pnl_pct)
            num_to_close = int(len(positions) * (1 - reduction_factor))
            symbols_to_close = [p.symbol for p in positions[:num_to_close]]
        
        for symbol in symbols_to_close:
            position = self.portfolio.positions.get(symbol)
            if position and symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                if not pd.isna(current_price):
                    exit_action = Action.SELL if position.quantity > 0 else Action.BUY
                    exit_trade = self.execution_engine.execute_market_order(
                        symbol, exit_action, abs(position.quantity), float(current_price), timestamp
                    )
                    self.portfolio.execute_trade(exit_trade)
                    self.portfolio.close_position(symbol, exit_trade, "risk_reduction")
    
    def get_results(self) -> Dict:
        """Get all backtest results."""
        from .analysis import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(
            self.portfolio.get_equity_dataframe(),
            self.portfolio.closed_positions,
            self.initial_capital
        )
        return {
            'metrics': metrics,
            'equity_curve': self.portfolio.get_equity_dataframe(),
            'drawdown_series': self.portfolio.get_drawdown_series(),
            'closed_positions': self.portfolio.closed_positions,
            'portfolio': self.portfolio,
            'audit_log': self.audit_log,
            'rejections': self.risk_engine.get_rejection_summary(),
        }
    
    def export_all_results(self, output_dir: str = '.'):
        """Export all results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Equity curve
        equity_df = self.portfolio.get_equity_dataframe()
        equity_df['drawdown'] = self.portfolio.get_drawdown_series()
        equity_df.to_csv(os.path.join(output_dir, 'equity_curve.csv'))
        
        # Trade log
        trade_log = self.portfolio.get_trade_log_dataframe()
        if not trade_log.empty:
            trade_log.to_csv(os.path.join(output_dir, 'trade_log.csv'), index=False)
        
        # Performance metrics
        from .analysis import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(equity_df, self.portfolio.closed_positions, self.initial_capital)
        pd.DataFrame([metrics.to_dict()]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Rejections
        rejections = self.risk_engine.get_rejection_summary()
        if not rejections.empty:
            rejections.to_csv(os.path.join(output_dir, 'rejections.csv'), index=False)
        
        self.logger.info(f"All results exported to {output_dir}")
