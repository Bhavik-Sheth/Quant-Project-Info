"""
Analysis Module - Performance Metrics & Walk-Forward Validation
Consolidated from performance.py + walk_forward.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .models import ClosedPosition, PortfolioState


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest"""
    # Returns
    total_return: float
    cagr: float
    avg_return_per_trade: float
    
    # Risk
    volatility: float
    max_drawdown: float
    downside_deviation: float
    value_at_risk_95: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    expectancy: float


class PerformanceAnalyzer:
    """Calculates comprehensive performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        closed_positions: List[ClosedPosition],
        initial_capital: float
    ) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        
        # Returns
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        years = len(equity_curve) / 252  # Assuming daily data
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        
        # Risk
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        drawdown_series = self._calculate_drawdown(equity_curve)
        max_drawdown = drawdown_series.min()
        
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        value_at_risk_95 = returns.quantile(0.05)
        
        # Risk-adjusted
        sharpe_ratio = (cagr - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        sortino_ratio = (cagr - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        # Trade statistics
        if len(closed_positions) > 0:
            trade_stats = self._calculate_trade_statistics(closed_positions)
        else:
            trade_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_return_per_trade': 0.0,
                'profit_factor': 0.0,
                'avg_holding_period': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'expectancy': 0.0
            }
        
        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            avg_return_per_trade=trade_stats['avg_return_per_trade'],
            volatility=volatility,
            max_drawdown=max_drawdown,
            downside_deviation=downside_deviation,
            value_at_risk_95=value_at_risk_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            avg_holding_period=trade_stats['avg_holding_period'],
            max_consecutive_wins=trade_stats['max_consecutive_wins'],
            max_consecutive_losses=trade_stats['max_consecutive_losses'],
            expectancy=trade_stats['expectancy']
        )
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        cumulative_max = equity_curve.expanding().max()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return drawdown
    
    def _calculate_trade_statistics(self, closed_positions: List[ClosedPosition]) -> Dict:
        """Calculate trade-level statistics"""
        if len(closed_positions) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_return_per_trade': 0.0,
                'profit_factor': 0.0,
                'avg_holding_period': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'expectancy': 0.0
            }
        
        returns = [pos.return_pct for pos in closed_positions]
        pnls = [pos.total_pnl for pos in closed_positions]
        
        total_trades = len(closed_positions)
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = sum(1 for r in returns if r < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_return_per_trade = np.mean(returns)
        
        total_wins = sum(p for p in pnls if p > 0)
        total_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        holding_periods = [(pos.exit_timestamp - pos.entry_timestamp).days for pos in closed_positions]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for r in returns:
            if r > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif r < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_return_per_trade': avg_return_per_trade,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'expectancy': expectancy
        }


# ==================== Walk-Forward Validation ====================

@dataclass
class WindowResult:
    """Results for a single walk-forward window"""
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    closed_positions: List[ClosedPosition]


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation"""
    window_results: List[WindowResult]
    aggregated_metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    consistency_score: float  # 0-1 score based on metric stability


class WalkForwardValidator:
    """
    Walk-forward validation for backtesting
    
    Splits data into multiple train/test windows to validate strategy robustness
    """
    
    def __init__(
        self,
        train_window_size: int,
        test_window_size: int,
        window_type: str = 'rolling',  # 'rolling' or 'expanding'
        step_size: int = None  # If None, defaults to test_window_size
    ):
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.window_type = window_type
        self.step_size = step_size or test_window_size
        
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_walk_forward(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        decision_generator: Callable,  # Function that generates trade decisions given train/test data
        initial_capital: float = 100000
    ) -> WalkForwardResults:
        """
        Run walk-forward validation
        
        Args:
            market_data: Market OHLCV data
            features: Feature data aligned with market data
            decision_generator: Function(train_data, train_features, test_data, test_features) -> List[TradeDecision]
            initial_capital: Starting capital
        """
        from .backtesting import BacktestEngine, ExecutionConfig, RiskConfig
        
        windows = self._create_windows(market_data.index.get_level_values(0).unique())
        window_results = []
        
        for idx, (train_period, test_period) in enumerate(windows):
            print(f"Processing window {idx + 1}/{len(windows)}...")
            
            # Split data
            train_data = market_data.loc[train_period]
            train_features = features.loc[train_period]
            test_data = market_data.loc[test_period]
            test_features = features.loc[test_period]
            
            # Generate decisions for this window
            trade_decisions = decision_generator(train_data, train_features, test_data, test_features)
            
            # Run backtest on test period
            engine = BacktestEngine(
                initial_capital=initial_capital,
                execution_config=ExecutionConfig(),
                risk_config=RiskConfig()
            )
            
            metrics = engine.run(test_data, test_features, trade_decisions)
            results = engine.get_results()
            
            window_result = WindowResult(
                window_index=idx,
                train_start=train_period[0],
                train_end=train_period[-1],
                test_start=test_period[0],
                test_end=test_period[-1],
                metrics=metrics,
                equity_curve=results['equity_curve'],
                closed_positions=results['closed_positions']
            )
            
            window_results.append(window_result)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(window_results)
        consistency_score = self._calculate_consistency_score(aggregated_metrics)
        
        return WalkForwardResults(
            window_results=window_results,
            aggregated_metrics=aggregated_metrics,
            consistency_score=consistency_score
        )
    
    def _create_windows(self, timestamps: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Create train/test window pairs"""
        timestamps = sorted(timestamps)
        windows = []
        
        start_idx = 0
        while start_idx + self.train_window_size + self.test_window_size <= len(timestamps):
            if self.window_type == 'rolling':
                train_start_idx = start_idx
            else:  # expanding
                train_start_idx = 0
            
            train_end_idx = start_idx + self.train_window_size
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_window_size
            
            train_period = timestamps[train_start_idx:train_end_idx]
            test_period = timestamps[test_start_idx:test_end_idx]
            
            windows.append((train_period, test_period))
            
            start_idx += self.step_size
        
        return windows
    
    def _aggregate_metrics(self, window_results: List[WindowResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across windows"""
        metric_names = [
            'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'volatility', 'win_rate', 'profit_factor'
        ]
        
        aggregated = {}
        for metric_name in metric_names:
            values = [getattr(wr.metrics, metric_name) for wr in window_results]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def _calculate_consistency_score(self, aggregated_metrics: Dict) -> float:
        """
        Calculate consistency score (0-1)
        
        Higher score = more consistent performance across windows
        """
        # Use coefficient of variation (CV) for key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'win_rate']
        
        cvs = []
        for metric in key_metrics:
            mean = aggregated_metrics[metric]['mean']
            std = aggregated_metrics[metric]['std']
            
            if mean != 0:
                cv = abs(std / mean)
                cvs.append(cv)
        
        avg_cv = np.mean(cvs) if cvs else 1.0
        
        # Convert to 0-1 score (lower CV = higher score)
        # CV of 0 = score of 1, CV of 1+ = score approaching 0
        consistency_score = 1 / (1 + avg_cv)
        
        return consistency_score
    
    def export_results(self, results: WalkForwardResults, output_dir: str):
        """Export walk-forward results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Window summary
        summary_data = []
        for wr in results.window_results:
            summary_data.append({
                'window_index': wr.window_index,
                'train_start': wr.train_start,
                'train_end': wr.train_end,
                'test_start': wr.test_start,
                'test_end': wr.test_end,
                'total_return': wr.metrics.total_return,
                'cagr': wr.metrics.cagr,
                'sharpe_ratio': wr.metrics.sharpe_ratio,
                'max_drawdown': wr.metrics.max_drawdown,
                'win_rate': wr.metrics.win_rate,
                'total_trades': wr.metrics.total_trades
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'window_summary.csv'), index=False)
        
        # Aggregated metrics
        agg_data = []
        for metric_name, stats in results.aggregated_metrics.items():
            agg_data.append({
                'metric': metric_name,
                **stats
            })
        
        agg_df = pd.DataFrame(agg_data)
        agg_df.to_csv(os.path.join(output_dir, 'aggregated_metrics.csv'), index=False)
        
        # Consistency score
        with open(os.path.join(output_dir, 'consistency_score.txt'), 'w') as f:
            f.write(f"Consistency Score: {results.consistency_score:.4f}\n")
        
        print(f"Walk-forward results exported to {output_dir}")
