"""
ML-Enhanced Strategy Wrapper

Uses ML model predictions to filter/enhance signals from base strategies.
Implements Decorator Pattern for composability.
"""

import sys
import os
from typing import Optional

# Add parent directory to path
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from quant_strategy.base import BaseStrategy, Signal, Action, Context, StrategyType


class MLSignalFilter(BaseStrategy):
    """
    ML-Enhanced Strategy Wrapper (Decorator Pattern).
    
    Wraps any BaseStrategy and filters signals based on ML confidence.
    
    Logic:
    1. Run the wrapped strategy's generate_signal()
    2. Check ML prediction confidence from context
    3. If confidence < threshold: Suppress signal (return HOLD)
    4. If confidence >= threshold: Pass through signal with ML reasoning appended
    
    Example:
        base_strategy = RSIStrategy()
        enhanced = MLSignalFilter(base_strategy, confidence_threshold=0.7)
        signal = enhanced.generate_signal(context)
    """
    
    def __init__(
        self,
        base_strategy: BaseStrategy,
        confidence_threshold: float = 0.70,
        ml_prediction_key: str = 'direction_confidence',
        name: Optional[str] = None
    ):
        """
        Initialize ML filter wrapper.
        
        Args:
            base_strategy: Underlying strategy to wrap
            confidence_threshold: Minimum ML confidence to allow signal (0-1)
            ml_prediction_key: Key in context.ml_predictions dict
            name: Optional custom name (defaults to "ML_{base_name}")
        """
        if name is None:
            name = f"ML_{base_strategy.name}"
        
        super().__init__(name, StrategyType.ML_ENHANCED)
        
        self.base_strategy = base_strategy
        self.confidence_threshold = confidence_threshold
        self.ml_prediction_key = ml_prediction_key
        
        self.parameters = {
            'confidence_threshold': confidence_threshold,
            'ml_prediction_key': ml_prediction_key
        }
    
    def generate_signal(self, context: Context) -> Signal:
        """
        Generate signal with ML filtering.
        
        Workflow:
        1. Get base strategy signal
        2. Extract ML confidence
        3. Filter or enhance based on ML agreement
        """
        # Step 1: Get base signal
        base_signal = self.base_strategy.generate_signal(context)
        
        # Step 2: Extract ML confidence
        if not context.ml_predictions:
            # No ML predictions available - pass through base signal
            return Signal(
                action=base_signal.action,
                confidence=base_signal.confidence * 0.8,  # Slight reduction
                reason=f"{base_signal.reason} [No ML confirmation available]",
                strategy_name=self.name,
                metadata=base_signal.metadata
            )
        
        ml_confidence = context.ml_predictions.get(self.ml_prediction_key, 0.0)
        
        # Step 3: Get ML direction prediction (if available)
        ml_direction = context.ml_predictions.get('predicted_direction', None)
        
        # Step 4: Filter logic
        
        # Case 1: ML confidence too low - suppress signal
        if ml_confidence < self.confidence_threshold:
            reason = (f"Base strategy ({self.base_strategy.name}) suggested {base_signal.action.value}, "
                     f"but ML confidence is low ({ml_confidence:.0%} < {self.confidence_threshold:.0%}). "
                     f"Signal suppressed for risk management.")
            
            return Signal(
                action=Action.HOLD,
                confidence=0.3,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'base_signal': base_signal.action.value,
                    'ml_confidence': ml_confidence,
                    'suppressed': True
                }
            )
        
        # Case 2: ML agrees with base signal - boost confidence
        if self._signals_agree(base_signal.action, ml_direction):
            boosted_confidence = min(0.95, base_signal.confidence * 1.25)
            
            reason = (f"{base_signal.reason} "
                     f"[ML CONFIRMATION: {ml_confidence:.0%} confidence agrees with {base_signal.action.value} signal]")
            
            return Signal(
                action=base_signal.action,
                confidence=boosted_confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    **base_signal.metadata,
                    'ml_confidence': ml_confidence,
                    'ml_agreement': True,
                    'confidence_boost': True
                }
            )
        
        # Case 3: ML disagrees but confidence is high enough - reduce confidence
        elif ml_direction is not None and not self._signals_agree(base_signal.action, ml_direction):
            reduced_confidence = base_signal.confidence * 0.6
            
            reason = (f"{base_signal.reason} "
                     f"[ML CONFLICT: Model predicts {ml_direction} with {ml_confidence:.0%} confidence, "
                     f"conflicting with {base_signal.action.value}. Proceeding with caution.]")
            
            return Signal(
                action=base_signal.action,
                confidence=reduced_confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    **base_signal.metadata,
                    'ml_confidence': ml_confidence,
                    'ml_direction': ml_direction,
                    'ml_conflict': True
                }
            )
        
        # Case 4: ML neutral or no direction - pass through with slight boost
        else:
            reason = (f"{base_signal.reason} "
                     f"[ML confidence: {ml_confidence:.0%}]")
            
            return Signal(
                action=base_signal.action,
                confidence=base_signal.confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    **base_signal.metadata,
                    'ml_confidence': ml_confidence
                }
            )
    
    def _signals_agree(self, action: Action, ml_direction: Optional[str]) -> bool:
        """
        Check if base signal agrees with ML direction prediction.
        
        Args:
            action: Base strategy action
            ml_direction: ML predicted direction ('UP', 'DOWN', etc.)
            
        Returns:
            True if signals agree
        """
        if ml_direction is None:
            return False
        
        ml_direction_upper = str(ml_direction).upper()
        
        # BUY agrees with UP/BULLISH
        if action == Action.BUY and ml_direction_upper in ['UP', 'BULLISH', 'LONG', 'BUY']:
            return True
        
        # SELL agrees with DOWN/BEARISH
        if action == Action.SELL and ml_direction_upper in ['DOWN', 'BEARISH', 'SHORT', 'SELL']:
            return True
        
        return False
    
    def adjust_parameters(self, volatility: float) -> None:
        """
        Adjust both wrapper and base strategy parameters.
        
        In high volatility, increase confidence threshold (more conservative).
        """
        super().adjust_parameters(volatility)
        
        # Adjust base strategy
        self.base_strategy.adjust_parameters(volatility)
        
        # Increase threshold in high volatility
        base_threshold = 0.70
        vol_adjustment = volatility * 5  # e.g., 2% vol = 0.10 increase
        self.parameters['confidence_threshold'] = min(0.90, base_threshold + vol_adjustment)
        self.confidence_threshold = self.parameters['confidence_threshold']


class MLDirectStrategy(BaseStrategy):
    """
    Pure ML-based strategy (no technical indicators).
    
    Generates signals directly from ML model predictions.
    Useful as a baseline or for comparison.
    """
    
    def __init__(
        self,
        name: str = "ML_Direct",
        min_confidence: float = 0.75
    ):
        """
        Initialize pure ML strategy.
        
        Args:
            name: Strategy identifier
            min_confidence: Minimum confidence to generate signal
        """
        super().__init__(name, StrategyType.ML_ENHANCED)
        self.parameters = {
            'min_confidence': min_confidence
        }
    
    def generate_signal(self, context: Context) -> Signal:
        """Generate signal purely from ML predictions."""
        if not context.ml_predictions:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No ML predictions available",
                strategy_name=self.name
            )
        
        ml_direction = context.ml_predictions.get('predicted_direction')
        ml_confidence = context.ml_predictions.get('direction_confidence', 0.0)
        
        if ml_confidence < self.parameters['min_confidence']:
            return Signal(
                action=Action.HOLD,
                confidence=ml_confidence,
                reason=f"ML confidence {ml_confidence:.0%} below threshold {self.parameters['min_confidence']:.0%}",
                strategy_name=self.name
            )
        
        # Convert ML direction to action
        if ml_direction and str(ml_direction).upper() in ['UP', 'BULLISH', 'LONG', 'BUY']:
            action = Action.BUY
            reason = f"ML model predicts upward movement with {ml_confidence:.0%} confidence"
        elif ml_direction and str(ml_direction).upper() in ['DOWN', 'BEARISH', 'SHORT', 'SELL']:
            action = Action.SELL
            reason = f"ML model predicts downward movement with {ml_confidence:.0%} confidence"
        else:
            action = Action.HOLD
            reason = f"ML model neutral or unclear (direction: {ml_direction})"
        
        return Signal(
            action=action,
            confidence=ml_confidence,
            reason=reason,
            strategy_name=self.name,
            metadata=context.ml_predictions
        )
