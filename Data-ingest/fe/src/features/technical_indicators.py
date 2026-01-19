"""
Feature engineering module for technical indicators.
Computes trend, momentum, volatility, and volume indicators.
All features are computed using only past data to prevent look-ahead bias.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes technical indicators for financial time series data.
    All calculations respect causality - only past data is used.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        pass
    
    # ==================== TREND INDICATORS ====================
    
    def sma(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for SMA
            column: Column to calculate SMA on
            
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=period, min_periods=period).mean()
    
    def ema(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for EMA
            column: Column to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    
    def macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = self.ema(df, fast)
        ema_slow = self.ema(df, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX).
        Measures trend strength.
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ADX values
        """
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the True Range and Directional Movements
        atr = tr.rolling(window=period, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        return adx
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for RSI
            column: Column to calculate RSI on
            
        Returns:
            Series with RSI values
        """
        delta = df[column].diff()
        
        gain = delta.where(delta > 0, 0)  # type: ignore
        loss = -delta.where(delta < 0, 0)  # type: ignore
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def roc(self, df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Rate of Change (ROC).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for ROC
            column: Column to calculate ROC on
            
        Returns:
            Series with ROC values
        """
        roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        return roc
    
    def stochastic_oscillator(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLC data
            k_period: Period for %K line
            d_period: Period for %D line (signal)
            
        Returns:
            Dictionary with %K and %D values
        """
        low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for moving average
            std_dev: Number of standard deviations
            column: Column to calculate bands on
            
        Returns:
            Dictionary with middle band, upper band, and lower band
        """
        middle_band = self.sma(df, period, column)
        std = df[column].rolling(window=period, min_periods=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_width': upper_band - lower_band
        }
    
    def historical_volatility(
        self,
        df: pd.DataFrame,
        period: int,
        column: str = 'close'
    ) -> pd.Series:
        """
        Historical Volatility (standard deviation of returns).
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate volatility on
            
        Returns:
            Series with historical volatility values
        """
        returns = df[column].pct_change()
        volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)
        
        return volatility
    
    # ==================== VOLUME INDICATORS ====================
    
    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP).
        Calculated as cumulative for each day.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # For intraday data, reset VWAP daily
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
            
            vwap_values = []
            for date in df_copy['date'].unique():
                day_data = df_copy[df_copy['date'] == date]
                day_typical = typical_price.loc[day_data.index]
                day_volume = df_copy.loc[day_data.index, 'volume']
                
                cum_tp_vol = (day_typical * day_volume).cumsum()
                cum_vol = day_volume.cumsum()  # type: ignore
                day_vwap = cum_tp_vol / cum_vol
                
                vwap_values.extend(day_vwap.values)
            
            return pd.Series(vwap_values, index=df.index)
        else:
            # For daily data, simple cumulative VWAP
            cum_tp_vol = (typical_price * df['volume']).cumsum()
            cum_vol = df['volume'].cumsum()
            return cum_tp_vol / cum_vol
    
    def obv(self, df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV).
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            Series with OBV values
        """
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def volume_roc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Volume Rate of Change.
        
        Args:
            df: DataFrame with volume data
            period: Number of periods
            
        Returns:
            Series with volume ROC values
        """
        return self.roc(df, period, column='volume')
    
    # ==================== FEATURE GENERATION ====================
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Generate all technical indicators based on configuration.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with indicator parameters
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        logger.info(f"Generating features for {len(df)} records")
        
        # Trend Indicators
        logger.info("Computing trend indicators...")
        for period in config.get('sma_periods', []):
            df[f'sma_{period}'] = self.sma(df, period)
        
        for period in config.get('ema_periods', []):
            df[f'ema_{period}'] = self.ema(df, period)
        
        macd_params = config.get('macd_params', {})
        if macd_params:
            macd_result = self.macd(
                df,
                fast=macd_params.get('fast', 12),
                slow=macd_params.get('slow', 26),
                signal=macd_params.get('signal', 9)
            )
            for key, value in macd_result.items():
                df[key] = value
        
        adx_period = config.get('adx_period', 14)
        if adx_period:
            df[f'adx_{adx_period}'] = self.adx(df, adx_period)
        
        # Momentum Indicators
        logger.info("Computing momentum indicators...")
        rsi_period = config.get('rsi_period', 14)
        if rsi_period:
            df[f'rsi_{rsi_period}'] = self.rsi(df, rsi_period)
        
        for period in config.get('roc_periods', []):
            df[f'roc_{period}'] = self.roc(df, period)
        
        stoch_params = config.get('stoch_params', {})
        if stoch_params:
            stoch_result = self.stochastic_oscillator(
                df,
                k_period=stoch_params.get('k_period', 14),
                d_period=stoch_params.get('d_period', 3)
            )
            for key, value in stoch_result.items():
                df[key] = value
        
        # Volatility Indicators
        logger.info("Computing volatility indicators...")
        atr_period = config.get('atr_period', 14)
        if atr_period:
            df[f'atr_{atr_period}'] = self.atr(df, atr_period)
        
        bb_params = config.get('bollinger_params', {})
        if bb_params:
            bb_result = self.bollinger_bands(
                df,
                period=bb_params.get('period', 20),
                std_dev=bb_params.get('std_dev', 2)
            )
            for key, value in bb_result.items():
                df[key] = value
        
        for period in config.get('hist_vol_periods', []):
            df[f'hist_vol_{period}'] = self.historical_volatility(df, period)
        
        # Volume Indicators
        logger.info("Computing volume indicators...")
        df['vwap'] = self.vwap(df)
        df['obv'] = self.obv(df)
        
        vol_roc_periods = config.get('roc_periods', [])
        for period in vol_roc_periods:
            df[f'volume_roc_{period}'] = self.volume_roc(df, period)
        
        logger.info(f"Feature generation complete. Total columns: {len(df.columns)}")
        
        return df
