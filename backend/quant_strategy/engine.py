"""
Backtesting Engine Module

MongoDB-based simulation loop with strict temporal integrity.
Ensures information at time t never leaks into execution at t+1.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

# MongoDB
from pymongo import MongoClient
from pymongo.collection import Collection

# Add parent directory to path for backend imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from quant_strategy.base import Context, Signal, Action, Regime, PortfolioState
from quant_strategy.components.ensemble import StrategyOrchestrator
from quant_strategy.components.risk_manager import RiskManager
from quant_strategy.utils import calculate_sharpe_ratio, calculate_max_drawdown

# Try to import ML models for regime/volatility predictions
try:
    from ML_Models.Regime_Classificaiton import Regime_Classifier
    from ML_Models.Volatility_Forecasting import Volatility_Models
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    print("Warning: ML_Models not available. Using fallback predictions.")


@dataclass
class Trade:
    """Record of a single trade execution."""
    timestamp: datetime
    symbol: str
    action: Action
    quantity: int
    price: float
    reason: str
    signal_confidence: float
    portfolio_value: float


class BacktestEngine:
    """
    Backtesting engine with MongoDB data source.
    
    Key Features:
    - Strict temporal integrity (t vs t+1 separation)
    - MongoDB batch fetching (memory efficient)
    - ML model integration for predictions
    - Strategy orchestration
    - Risk management gates
    - Performance tracking
    """
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "trading_db",
        collection_name: str = "market_data",
        initial_capital: float = 100000.0
    ):
        """
        Initialize backtesting engine.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            collection_name: Collection with market data
            initial_capital: Starting portfolio value
        """
        # MongoDB connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        
        # Portfolio
        self.initial_capital = initial_capital
        self.portfolio: Optional[PortfolioState] = None
        
        # Risk management
        self.risk_manager = RiskManager()
        
        # Trade log
        self.trades: List[Trade] = []
        
        # ML models (initialized on first use)
        self.regime_model: Optional[Any] = None
        self.volatility_model: Optional[Any] = None
    
    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        orchestrator: StrategyOrchestrator,
        symbol: str = "AAPL",
        batch_size: int = 1000,
        default_quantity: int = 100
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        CRITICAL WORKFLOW:
        1. Fetch data at time t
        2. Build Context with info available at t
        3. Generate Signal at t
        4. Execute trade at t+1 (NEXT bar's open price)
        
        Args:
            start_date: Backtest start
            end_date: Backtest end
            orchestrator: Strategy orchestrator
            symbol: Trading symbol
            batch_size: MongoDB batch size
            default_quantity: Default shares per trade
            
        Returns:
            Performance metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"STARTING BACKTEST: {symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        
        # Initialize portfolio
        self.portfolio = PortfolioState(
            cash=self.initial_capital,
            positions={symbol: 0},
            total_value=self.initial_capital
        )
        
        # Fetch data generator
        data_cursor = self._fetch_data(symbol, start_date, end_date, batch_size)
        
        # Convert to list for lookahead (need t+1 for execution)
        data_bars = list(data_cursor)
        
        if len(data_bars) < 2:
            print("Insufficient data for backtest")
            return {'error': 'Insufficient data'}
        
        print(f"Loaded {len(data_bars)} data bars\n")
        
        # Main simulation loop
        for i in range(len(data_bars) - 1):  # -1 because we need t+1
            current_bar = data_bars[i]   # Time t
            next_bar = data_bars[i + 1]  # Time t+1
            
            # ===== TIME t: BUILD CONTEXT =====
            context = self._build_context(current_bar, symbol)
            
            if context is None:
                continue  # Skip if context building failed
            
            # ===== TIME t: GENERATE SIGNAL =====
            signal = orchestrator.decide(context)
            
            # ===== TIME t: RISK CHECK =====
            if signal.action != Action.HOLD:
                risk_ok, risk_reasons = self.risk_manager.validate_signal(
                    context, signal, default_quantity
                )
                
                if not risk_ok:
                    print(f"[{context.timestamp.date()}] RISK BLOCK: {risk_reasons[0]}")
                    signal = Signal(
                        action=Action.HOLD,
                        confidence=0.0,
                        reason=f"Risk blocked: {risk_reasons[0]}",
                        strategy_name="RiskManager"
                    )
            
            # ===== TIME t+1: EXECUTE =====
            execution_price = next_bar.get('open', next_bar.get('close'))
            
            if signal.action != Action.HOLD:
                self._execute_trade(
                    signal, context, execution_price, default_quantity
                )
            
            # Update portfolio value
            current_prices = {symbol: current_bar['close']}
            self.portfolio.update_value(current_prices)
            
            # Log progress periodically
            if i % 50 == 0:
                print(f"[{context.timestamp.date()}] Portfolio: ${self.portfolio.total_value:,.2f} | "
                      f"Position: {self.portfolio.get_position(symbol)} shares | "
                      f"Signal: {signal.action.value}")
        
        # Final performance metrics
        metrics = self._calculate_performance()
        
        print(f"\n{'='*60}")
        print(f"BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"Final Value: ${self.portfolio.total_value:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch data from MongoDB in batches.
        
        Returns generator to avoid loading all data into memory.
        """
        cursor = self.collection.find({
            'symbol': symbol,
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }).sort('timestamp', 1).batch_size(batch_size)
        
        for document in cursor:
            yield document
    
    def _build_context(self, data_bar: Dict[str, Any], symbol: str) -> Optional[Context]:
        """
        Build Context from MongoDB document.
        
        CRITICAL: Only use data available at time t!
        """
        try:
            timestamp = data_bar['timestamp']
            price = data_bar['close']
            
            # Extract features (technical indicators)
            features = self._extract_features(data_bar)
            
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(data_bar, features)
            
            # Detect regime
            current_regime = self._detect_regime(ml_predictions, features)
            
            # Build context
            context = Context(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                features=features,
                ml_predictions=ml_predictions,
                portfolio=self.portfolio,
                option_chain=None,  # TODO: Add if available
                current_regime=current_regime
            )
            
            return context
            
        except Exception as e:
            print(f"Error building context: {e}")
            return None
    
    def _extract_features(self, data_bar: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical indicators from data bar."""
        features = {}
        
        # Common feature keys from Data-inges-fe
        feature_keys = [
            'RSI', 'RSI_14', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR', 'ATR_14',
            'Stoch_K', 'Stoch_D', 'ADX', 'OBV', 'VWAP'
        ]
        
        for key in feature_keys:
            if key in data_bar:
                try:
                    features[key] = float(data_bar[key])
                except (ValueError, TypeError):
                    pass
        
        # OHLCV
        for key in ['open', 'high', 'low', 'close', 'volume']:
            if key in data_bar:
                try:
                    features[key] = float(data_bar[key])
                except (ValueError, TypeError):
                    pass
        
        return features
    
    def _get_ml_predictions(
        self,
        data_bar: Dict[str, Any],
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Get ML model predictions for volatility, direction, etc.
        
        Uses ML_Models if available, otherwise returns fallback predictions.
        """
        predictions = {}
        
        if not ML_MODELS_AVAILABLE:
            # Fallback: Use historical volatility
            if 'ATR' in features and 'close' in features:
                predictions['predicted_volatility'] = features['ATR'] / features['close']
            else:
                predictions['predicted_volatility'] = 0.02  # Default 2%
            
            predictions['direction_confidence'] = 0.5
            predictions['predicted_direction'] = 'NEUTRAL'
            
            return predictions
        
        # TODO: Integrate actual ML models
        # For now, use fallback
        predictions['predicted_volatility'] = features.get('ATR', 0) / features.get('close', 1) if features.get('close', 0) > 0 else 0.02
        predictions['direction_confidence'] = 0.5
        predictions['predicted_direction'] = 'NEUTRAL'
        
        return predictions
    
    def _detect_regime(
        self,
        ml_predictions: Dict[str, Any],
        features: Dict[str, float]
    ) -> Regime:
        """
        Detect market regime using ML model or heuristics.
        
        Fallback heuristic:
        - High volatility + trending = VOLATILE
        - Low volatility + ranging = RANGING
        - Strong trend + low vol = BULLISH/BEARISH
        """
        # Simple heuristic regime detection
        volatility = ml_predictions.get('predicted_volatility', 0.02)
        
        # Check for trend using MAs
        sma_50 = features.get('SMA_50')
        sma_200 = features.get('SMA_200')
        current_price = features.get('close')
        
        if volatility > 0.03:  # High volatility
            return Regime.VOLATILE
        
        if sma_50 and sma_200 and current_price:
            if sma_50 > sma_200 * 1.02:  # Uptrend
                return Regime.BULLISH
            elif sma_50 < sma_200 * 0.98:  # Downtrend
                return Regime.BEARISH
        
        return Regime.RANGING
    
    def _execute_trade(
        self,
        signal: Signal,
        context: Context,
        execution_price: float,
        quantity: int
    ) -> None:
        """
        Execute trade at t+1 price.
        
        Updates portfolio and logs trade.
        """
        symbol = context.symbol
        
        if signal.action == Action.BUY:
            cost = quantity * execution_price
            
            if self.portfolio.cash >= cost:
                self.portfolio.cash -= cost
                self.portfolio.positions[symbol] = self.portfolio.get_position(symbol) + quantity
                
                print(f"[{context.timestamp.date()}] BUY {quantity} @ ${execution_price:.2f} | {signal.reason[:80]}...")
                
                self.trades.append(Trade(
                    timestamp=context.timestamp,
                    symbol=symbol,
                    action=Action.BUY,
                    quantity=quantity,
                    price=execution_price,
                    reason=signal.reason,
                    signal_confidence=signal.confidence,
                    portfolio_value=self.portfolio.total_value
                ))
        
        elif signal.action == Action.SELL:
            current_position = self.portfolio.get_position(symbol)
            
            if current_position >= quantity:
                proceeds = quantity * execution_price
                self.portfolio.cash += proceeds
                self.portfolio.positions[symbol] = current_position - quantity
                
                print(f"[{context.timestamp.date()}] SELL {quantity} @ ${execution_price:.2f} | {signal.reason[:80]}...")
                
                self.trades.append(Trade(
                    timestamp=context.timestamp,
                    symbol=symbol,
                    action=Action.SELL,
                    quantity=quantity,
                    price=execution_price,
                    reason=signal.reason,
                    signal_confidence=signal.confidence,
                    portfolio_value=self.portfolio.total_value
                ))
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio or not self.portfolio.equity_curve:
            return {}
        
        equity_curve = self.portfolio.equity_curve
        
        # Calculate returns
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Metrics
        total_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        sharpe = calculate_sharpe_ratio(returns.tolist()) if len(returns) > 0 else 0.0
        max_dd = calculate_max_drawdown(equity_curve)
        
        # Trade stats
        winning_trades = [t for t in self.trades if self._is_winning_trade(t)]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'final_value': self.portfolio.total_value,
            'equity_curve': equity_curve
        }
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable (simplified)."""
        # TODO: Implement proper P&L tracking
        return trade.signal_confidence > 0.7
    
    def export_trades(self, filepath: str) -> None:
        """Export trade log to CSV."""
        if not self.trades:
            print("No trades to export")
            return
        
        trades_data = [
            {
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'action': t.action.value,
                'quantity': t.quantity,
                'price': t.price,
                'confidence': t.signal_confidence,
                'portfolio_value': t.portfolio_value,
                'reason': t.reason
            }
            for t in self.trades
        ]
        
        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        print(f"Trades exported to {filepath}")
