"""
Options Trading Strategy

Volatility arbitrage through options mispricing detection.
Compares model-predicted volatility with implied volatility.
"""

import sys
import os
from typing import Dict, Optional, Any

# Add parent directory to path
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from quant_strategy.base import BaseStrategy, Signal, Action, Context, StrategyType
from quant_strategy.utils import black_scholes


class VolArbitrageStrategy(BaseStrategy):
    """
    Volatility Arbitrage via Options Mispricing.
    
    Logic:
    1. Get implied volatility from market (option_chain)
    2. Get predicted volatility from ML model (GARCH/LSTM)
    3. Calculate fair value using Black-Scholes with predicted vol
    4. Compare fair value vs market price
    5. Signal if mispricing > threshold
    
    Integration:
    - Uses Volatility_Models from ML_Models/Volatility_Forecasting.py
    - Requires option_chain data in Context
    """
    
    def __init__(
        self,
        name: str = "VolArbitrage",
        mispricing_threshold: float = 0.10,
        risk_free_rate: float = 0.04,
        option_type: str = "call",
        min_vol_differential: float = 0.05
    ):
        """
        Initialize volatility arbitrage strategy.
        
        Args:
            name: Strategy identifier
            mispricing_threshold: Min % mispricing to signal (default 10%)
            risk_free_rate: Annual risk-free rate (default 4%)
            option_type: 'call' or 'put'
            min_vol_differential: Min vol spread to consider (default 5%)
        """
        super().__init__(name, StrategyType.OPTIONS)
        
        self.parameters = {
            'mispricing_threshold': mispricing_threshold,
            'risk_free_rate': risk_free_rate,
            'option_type': option_type,
            'min_vol_differential': min_vol_differential
        }
    
    def generate_signal(self, context: Context) -> Signal:
        """
        Generate signal based on options mispricing.
        
        Returns BUY if option undervalued, SELL if overvalued.
        """
        # Step 1: Validate requirements
        if context.option_chain is None or not context.option_chain:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No option chain data available for volatility arbitrage",
                strategy_name=self.name
            )
        
        if 'predicted_volatility' not in context.ml_predictions:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No volatility prediction from ML model",
                strategy_name=self.name
            )
        
        # Step 2: Extract data
        predicted_vol = context.ml_predictions['predicted_volatility']
        
        # Get ATM (at-the-money) option or closest strike
        option_data = self._find_atm_option(context.option_chain, context.price)
        
        if option_data is None:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No suitable options found in chain",
                strategy_name=self.name
            )
        
        # Step 3: Extract option parameters
        strike = option_data['strike']
        market_price = option_data['market_price']
        implied_vol = option_data.get('implied_vol')
        time_to_expiry = option_data.get('time_to_expiry', 30 / 365)  # Default 30 days
        
        # Step 4: Calculate fair value using Black-Scholes
        fair_value = black_scholes(
            S=context.price,
            K=strike,
            T=time_to_expiry,
            r=self.parameters['risk_free_rate'],
            sigma=predicted_vol,
            option_type=self.parameters['option_type']
        )
        
        # Step 5: Calculate mispricing
        mispricing_pct = (fair_value - market_price) / market_price if market_price > 0 else 0.0
        
        # Step 6: Check volatility differential
        if implied_vol is not None:
            vol_spread = abs(predicted_vol - implied_vol)
            
            if vol_spread < self.parameters['min_vol_differential']:
                reason = (f"Volatility spread too small: Predicted {predicted_vol:.1%} vs "
                         f"Implied {implied_vol:.1%} = {vol_spread:.1%} spread "
                         f"(min: {self.parameters['min_vol_differential']:.1%})")
                return Signal(
                    action=Action.HOLD,
                    confidence=0.4,
                    reason=reason,
                    strategy_name=self.name,
                    metadata={
                        'predicted_vol': predicted_vol,
                        'implied_vol': implied_vol,
                        'vol_spread': vol_spread
                    }
                )
        
        # Step 7: Generate signal based on mispricing
        threshold = self.parameters['mispricing_threshold']
        
        # Undervalued - BUY
        if mispricing_pct > threshold:
            confidence = min(0.90, 0.6 + abs(mispricing_pct) * 2)
            
            vol_info = f" IV: {implied_vol:.1%}" if implied_vol else ""
            reason = (f"Option undervalued by {mispricing_pct:.1%}. "
                     f"Fair Value: ${fair_value:.2f} vs Market: ${market_price:.2f}. "
                     f"Using predicted vol {predicted_vol:.1%}{vol_info}. "
                     f"Strike: ${strike:.2f}, Expiry: {time_to_expiry*365:.0f} days")
            
            return Signal(
                action=Action.BUY,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'fair_value': fair_value,
                    'market_price': market_price,
                    'mispricing_pct': mispricing_pct,
                    'predicted_vol': predicted_vol,
                    'implied_vol': implied_vol,
                    'strike': strike
                }
            )
        
        # Overvalued - SELL
        elif mispricing_pct < -threshold:
            confidence = min(0.90, 0.6 + abs(mispricing_pct) * 2)
            
            vol_info = f" IV: {implied_vol:.1%}" if implied_vol else ""
            reason = (f"Option overvalued by {abs(mispricing_pct):.1%}. "
                     f"Fair Value: ${fair_value:.2f} vs Market: ${market_price:.2f}. "
                     f"Using predicted vol {predicted_vol:.1%}{vol_info}. "
                     f"Strike: ${strike:.2f}, Expiry: {time_to_expiry*365:.0f} days")
            
            return Signal(
                action=Action.SELL,
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'fair_value': fair_value,
                    'market_price': market_price,
                    'mispricing_pct': mispricing_pct,
                    'predicted_vol': predicted_vol,
                    'implied_vol': implied_vol,
                    'strike': strike
                }
            )
        
        # Fair priced - HOLD
        else:
            reason = (f"Option fairly priced (mispricing: {mispricing_pct:.1%}, "
                     f"threshold: ±{threshold:.1%}). Fair: ${fair_value:.2f}, "
                     f"Market: ${market_price:.2f}")
            
            return Signal(
                action=Action.HOLD,
                confidence=0.5,
                reason=reason,
                strategy_name=self.name,
                metadata={
                    'fair_value': fair_value,
                    'market_price': market_price,
                    'mispricing_pct': mispricing_pct
                }
            )
    
    def _find_atm_option(
        self,
        option_chain: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find at-the-money (ATM) option or closest strike.
        
        Args:
            option_chain: Options data dictionary
            current_price: Current underlying price
            
        Returns:
            Option data dict or None
        """
        # Option chain format: {strike: {data}}
        if not option_chain:
            return None
        
        # Find closest strike to current price
        strikes = []
        for key in option_chain.keys():
            try:
                strike = float(key)
                strikes.append(strike)
            except (ValueError, TypeError):
                continue
        
        if not strikes:
            return None
        
        # Get ATM strike (closest to current price)
        atm_strike = min(strikes, key=lambda s: abs(s - current_price))
        
        option_data = option_chain[str(atm_strike)]
        
        # Ensure required fields
        if 'market_price' not in option_data:
            return None
        
        # Add strike if not present
        if 'strike' not in option_data:
            option_data['strike'] = atm_strike
        
        return option_data
    
    def adjust_parameters(self, volatility: float) -> None:
        """
        Adjust strategy parameters for volatility regime.
        
        In high volatility, increase mispricing threshold (be more selective).
        """
        super().adjust_parameters(volatility)
        
        base_threshold = 0.10
        # Increase threshold in high vol (avoid false signals)
        vol_multiplier = 1 + volatility * 5
        self.parameters['mispricing_threshold'] = base_threshold * vol_multiplier
