"""
Technical Analysis Strategies

Pure technical indicator-based strategies:
- RSI Strategy (oversold/overbought)
- Moving Average Crossover (trend following)
- Bollinger Bands (mean reversion)
"""

import sys
import os
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from quant_strategy.base import BaseStrategy, Signal, Action, Context, StrategyType


class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Mean Reversion Strategy.
    
    Logic:
    - BUY when RSI < oversold_threshold (default 30) - Asset is oversold
    - SELL when RSI > overbought_threshold (default 70) - Asset is overbought
    - HOLD otherwise
    """
    
    def __init__(
        self,
        name: str = "RSIStrategy",
        oversold: float = 30.0,
        overbought: float = 70.0,
        rsi_period: int = 14
    ):
        """
        Initialize RSI strategy.
        
        Args:
            name: Strategy identifier
            oversold: RSI level considered oversold (default 30)
            overbought: RSI level considered overbought (default 70)
            rsi_period: RSI calculation period (default 14)
        """
        super().__init__(name, StrategyType.TECHNICAL)
        self.parameters = {
            'oversold': oversold,
            'overbought': overbought,
            'rsi_period': rsi_period,
            'base_oversold': oversold,
            'base_overbought': overbought
        }
    
    def generate_signal(self, context: Context) -> Signal:
        """Generate signal based on RSI levels."""
        # Extract RSI from features
        rsi_key = f'RSI_{self.parameters["rsi_period"]}'
        if rsi_key not in context.features:
            # Try common alternatives
            if 'RSI' in context.features:
                rsi = context.features['RSI']
            elif 'rsi' in context.features:
                rsi = context.features['rsi']
            else:
                return Signal(
                    action=Action.HOLD,
                    confidence=0.0,
                    reason=f"RSI indicator not found in features (looking for {rsi_key})",
                    strategy_name=self.name
                )
        else:
            rsi = context.features[rsi_key]
        
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        
        # Generate signal
        if rsi < oversold:
            # Oversold - BUY signal
            confidence = (oversold - rsi) / oversold  # Stronger signal as RSI drops
            confidence = min(0.95, max(0.5, confidence))
            
            reason = (f"RSI at {rsi:.1f} is below oversold threshold of {oversold:.1f}, "
                     f"indicating potential mean reversion opportunity. "
                     f"Asset appears oversold.")
            
            return Signal(
                action=Action.BUY,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={'rsi': rsi, 'threshold': oversold}
            )
        
        elif rsi > overbought:
            # Overbought - SELL signal
            confidence = (rsi - overbought) / (100 - overbought)
            confidence = min(0.95, max(0.5, confidence))
            
            reason = (f"RSI at {rsi:.1f} is above overbought threshold of {overbought:.1f}, "
                     f"indicating potential reversal. Asset appears overbought.")
            
            return Signal(
                action=Action.SELL,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={'rsi': rsi, 'threshold': overbought}
            )
        
        else:
            # Neutral zone - HOLD
            reason = (f"RSI at {rsi:.1f} is in neutral zone "
                     f"(between {oversold:.1f} and {overbought:.1f})")
            
            return Signal(
                action=Action.HOLD,
                confidence=0.5,
                reason=reason,
                strategy_name=self.name,
                metadata={'rsi': rsi}
            )
    
    def adjust_parameters(self, volatility: float) -> None:
        """Widen RSI thresholds in high volatility to avoid whipsaws."""
        super().adjust_parameters(volatility)
        
        base_oversold = self.parameters['base_oversold']
        base_overbought = self.parameters['base_overbought']
        
        # In high volatility, widen the bands (more extreme RSI needed)
        vol_adjustment = volatility * 50  # e.g., 2% vol = 1 point adjustment
        
        self.parameters['oversold'] = max(20, base_oversold - vol_adjustment)
        self.parameters['overbought'] = min(80, base_overbought + vol_adjustment)


class MovingAverageCrossStrategy(BaseStrategy):
    """
    Moving Average Crossover Trend Following Strategy.
    
    Logic:
    - BUY when fast MA crosses above slow MA (Golden Cross)
    - SELL when fast MA crosses below slow MA (Death Cross)
    - HOLD otherwise
    """
    
    def __init__(
        self,
        name: str = "MA_Cross",
        fast_period: int = 50,
        slow_period: int = 200
    ):
        """
        Initialize MA crossover strategy.
        
        Args:
            name: Strategy identifier
            fast_period: Fast MA period (default 50)
            slow_period: Slow MA period (default 200)
        """
        super().__init__(name, StrategyType.TECHNICAL)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        self.last_cross_state: Optional[str] = None
    
    def generate_signal(self, context: Context) -> Signal:
        """Generate signal based on MA crossover."""
        fast_key = f'SMA_{self.parameters["fast_period"]}'
        slow_key = f'SMA_{self.parameters["slow_period"]}'
        
        # Try to find MAs in features
        fast_ma = context.features.get(fast_key) or context.features.get('SMA_50')
        slow_ma = context.features.get(slow_key) or context.features.get('SMA_200')
        
        if fast_ma is None or slow_ma is None:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason=f"Moving averages not found (need {fast_key} and {slow_key})",
                strategy_name=self.name
            )
        
        # Current state
        current_price = context.price
        
        # Check crossover
        if fast_ma > slow_ma:
            current_state = "GOLDEN"
        elif fast_ma < slow_ma:
            current_state = "DEATH"
        else:
            current_state = "NEUTRAL"
        
        # Detect crossover (change in state)
        if self.last_cross_state is None:
            self.last_cross_state = current_state
            return Signal(
                action=Action.HOLD,
                confidence=0.5,
                reason=f"Initializing: Fast MA ({fast_ma:.2f}) vs Slow MA ({slow_ma:.2f})",
                strategy_name=self.name
            )
        
        # Golden Cross
        if current_state == "GOLDEN" and self.last_cross_state != "GOLDEN":
            self.last_cross_state = current_state
            
            # Calculate distance between MAs for confidence
            spread_pct = (fast_ma - slow_ma) / slow_ma
            confidence = min(0.90, 0.7 + spread_pct * 10)
            
            reason = (f"Golden Cross detected: SMA{self.parameters['fast_period']} "
                     f"({fast_ma:.2f}) crossed above SMA{self.parameters['slow_period']} "
                     f"({slow_ma:.2f}). Bullish trend confirmed.")
            
            return Signal(
                action=Action.BUY,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        
        # Death Cross
        elif current_state == "DEATH" and self.last_cross_state != "DEATH":
            self.last_cross_state = current_state
            
            spread_pct = (slow_ma - fast_ma) / slow_ma
            confidence = min(0.90, 0.7 + spread_pct * 10)
            
            reason = (f"Death Cross detected: SMA{self.parameters['fast_period']} "
                     f"({fast_ma:.2f}) crossed below SMA{self.parameters['slow_period']} "
                     f"({slow_ma:.2f}). Bearish trend confirmed.")
            
            return Signal(
                action=Action.SELL,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        
        else:
            # No new crossover
            self.last_cross_state = current_state
            
            return Signal(
                action=Action.HOLD,
                confidence=0.5,
                reason=f"No crossover. Current state: {current_state}",
                strategy_name=self.name,
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Logic:
    - BUY when price touches/breaks below lower band (oversold)
    - SELL when price touches/breaks above upper band (overbought)
    - HOLD when price is within bands
    """
    
    def __init__(
        self,
        name: str = "BollingerBands",
        period: int = 20,
        num_std: float = 2.0
    ):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            name: Strategy identifier
            period: MA period for middle band (default 20)
            num_std: Number of standard deviations (default 2.0)
        """
        super().__init__(name, StrategyType.TECHNICAL)
        self.parameters = {
            'period': period,
            'num_std': num_std,
            'base_num_std': num_std
        }
    
    def generate_signal(self, context: Context) -> Signal:
        """Generate signal based on Bollinger Band position."""
        # Look for Bollinger Bands in features
        bb_upper = context.features.get('BB_Upper') or context.features.get('BB_UPPER')
        bb_lower = context.features.get('BB_Lower') or context.features.get('BB_LOWER')
        bb_middle = context.features.get('BB_Middle') or context.features.get('SMA_20')
        
        if bb_upper is None or bb_lower is None:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="Bollinger Bands indicators not found in features",
                strategy_name=self.name
            )
        
        current_price = context.price
        
        # Calculate position within bands
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="Bollinger Bands width is zero",
                strategy_name=self.name
            )
        
        # Price below lower band - Oversold
        if current_price < bb_lower:
            # How far below?
            distance_pct = (bb_lower - current_price) / band_width
            confidence = min(0.90, 0.6 + distance_pct * 2)
            
            reason = (f"Price ${current_price:.2f} is below lower Bollinger Band "
                     f"(${bb_lower:.2f}), suggesting oversold conditions. "
                     f"Mean reversion opportunity.")
            
            return Signal(
                action=Action.BUY,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'price': current_price,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'distance_from_band': distance_pct
                }
            )
        
        # Price above upper band - Overbought
        elif current_price > bb_upper:
            distance_pct = (current_price - bb_upper) / band_width
            confidence = min(0.90, 0.6 + distance_pct * 2)
            
            reason = (f"Price ${current_price:.2f} is above upper Bollinger Band "
                     f"(${bb_upper:.2f}), suggesting overbought conditions. "
                     f"Potential reversal.")
            
            return Signal(
                action=Action.SELL,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'price': current_price,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'distance_from_band': distance_pct
                }
            )
        
        # Price within bands - HOLD
        else:
            position = (current_price - bb_lower) / band_width  # 0-1
            
            reason = (f"Price ${current_price:.2f} is within Bollinger Bands "
                     f"(${bb_lower:.2f} - ${bb_upper:.2f}), no extreme detected")
            
            return Signal(
                action=Action.HOLD,
                confidence=0.5,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'price': current_price,
                    'bb_position': position,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper
                }
            )
    
    def adjust_parameters(self, volatility: float) -> None:
        """Widen bands in high volatility environments."""
        super().adjust_parameters(volatility)
        
        base_std = self.parameters['base_num_std']
        
        # Increase std multiplier in high vol to avoid false signals
        vol_multiplier = 1 + volatility * 10  # e.g., 2% vol = 1.2x
        self.parameters['num_std'] = base_std * vol_multiplier
