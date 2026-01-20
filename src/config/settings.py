"""
Configuration settings for the financial data pipeline.
All timestamps are normalized to UTC.
"""

from typing import List, Dict
from datetime import datetime, timedelta

# Data Source Configuration
DATA_SOURCE = "Yahoo Finance"

# Supported Timeframes
TIMEFRAMES = ["1m", "5m", "1h", "1d"]

# Default Symbols to Fetch - Focus on Liquid Instruments

# Major Indices (Core Assets)
DEFAULT_INDEX_SYMBOLS = [
    "^GSPC",   # S&P 500 (US Large Cap)
    "^IXIC",   # NASDAQ Composite (US Tech)
    "^DJI",    # Dow Jones Industrial Average (US Blue Chip)
    "^NSEI",   # NIFTY 50 (India)
    "^NSEBANK",# BANKNIFTY (India Banking)
    "^VIX",    # CBOE Volatility Index
]

# Liquid Individual Stocks - Diversified Across Sectors (30 stocks)
DEFAULT_EQUITY_SYMBOLS = [
    # Technology (7 stocks)
    "AAPL",    # Apple Inc. - Consumer Electronics
    "MSFT",    # Microsoft - Software/Cloud
    "GOOGL",   # Alphabet - Search/Advertising
    "AMZN",    # Amazon - E-commerce/Cloud
    "NVDA",    # NVIDIA - Semiconductors/AI
    "META",    # Meta Platforms - Social Media
    "ORCL",    # Oracle - Enterprise Software
    
    # Financial Services (5 stocks)
    "JPM",     # JPMorgan Chase - Banking
    "BAC",     # Bank of America - Banking
    "GS",      # Goldman Sachs - Investment Banking
    "V",       # Visa - Payments
    "MA",      # Mastercard - Payments
    
    # Healthcare (4 stocks)
    "JNJ",     # Johnson & Johnson - Pharmaceuticals
    "UNH",     # UnitedHealth - Health Insurance
    "PFE",     # Pfizer - Pharmaceuticals
    "ABBV",    # AbbVie - Biotechnology
    
    # Consumer (4 stocks)
    "WMT",     # Walmart - Retail
    "PG",      # Procter & Gamble - Consumer Goods
    "KO",      # Coca-Cola - Beverages
    "MCD",     # McDonald's - Fast Food
    
    # Energy (3 stocks)
    "XOM",     # Exxon Mobil - Oil & Gas
    "CVX",     # Chevron - Oil & Gas
    "COP",     # ConocoPhillips - Oil & Gas
    
    # Industrials (3 stocks)
    "BA",      # Boeing - Aerospace
    "CAT",     # Caterpillar - Heavy Machinery
    "GE",      # General Electric - Conglomerate
    
    # Communication Services (2 stocks)
    "DIS",     # Disney - Media/Entertainment
    "NFLX",    # Netflix - Streaming
    
    # Automotive/EVs (2 stocks)
    "TSLA",    # Tesla - Electric Vehicles
    "F",       # Ford - Automotive
]

# Date Range Configuration
# Extended lookback periods for comprehensive historical analysis
# Capturing multiple market regimes: crashes, rallies, sideways periods
DEFAULT_LOOKBACK_DAYS = {
    "1m": 7,      # Yahoo Finance limits 1m data to 7 days
    "5m": 60,     # Yahoo Finance limits 5m data to 60 days
    "1h": 730,    # 2 years for intraday patterns
    "1d": 3650,   # 10 years - captures 2008 crisis, 2020 COVID crash, recovery periods
}

# Validation Thresholds
VALIDATION_CONFIG = {
    "price_outlier_threshold": 0.20,  # 20% price change threshold
    "max_gap_multiplier": 2.0,        # Max gap = 2x expected interval
}

# Feature Engineering Configuration - ~60 Technical Indicators
FEATURE_CONFIG = {
    # ==================== TREND INDICATORS (~20 features) ====================
    # Simple Moving Averages - Support/Resistance levels
    "sma_periods": [10, 20, 50, 100, 200],  # 5 SMAs
    
    # Exponential Moving Averages - Responsive to recent changes
    "ema_periods": [12, 26, 50],  # 3 EMAs
    
    # MACD - Trend changes and momentum shifts
    "macd_params": {
        "fast": 12,
        "slow": 26,
        "signal": 9,
    },  # 3 features: MACD line, signal line, histogram
    
    # ADX - Trend strength quantification (crucial for regime detection)
    "adx_period": 14,  # 1 feature
    
    # Moving Average Crossovers and Relationships
    "ma_crossovers": True,  # ~8 features: price vs MA, MA crossovers
    
    # ==================== MOMENTUM INDICATORS (~15 features) ====================
    # RSI - Overbought/Oversold conditions (mean reversion)
    "rsi_period": 14,  # 1 feature (0-100 scale)
    
    # Multi-period RSI for different timeframes
    "rsi_periods_multi": [7, 14, 21],  # 3 RSI variants
    
    # Stochastic Oscillator - Price position in range
    "stoch_params": {
        "k_period": 14,
        "d_period": 3,
    },  # 2 features: %K and %D
    
    # Rate of Change - Momentum strength at multiple periods
    "roc_periods": [5, 10, 20],  # 3 ROC features
    
    # Commodity Channel Index (CCI) - Overbought/Oversold
    "cci_period": 20,  # 1 feature
    
    # Money Flow Index (MFI) - Volume-weighted RSI
    "mfi_period": 14,  # 1 feature
    
    # Ultimate Oscillator - Multi-timeframe momentum
    "ultimate_osc_periods": [7, 14, 28],  # 1 feature (combines 3 periods)
    
    # ==================== VOLATILITY INDICATORS (~12 features) ====================
    # Bollinger Bands - Price envelope (95% containment)
    "bollinger_params": {
        "period": 20,
        "std_dev": 2,
    },  # 4 features: middle, upper, lower, bandwidth
    
    # ATR - Risk measurement for position sizing and stops
    "atr_period": 14,  # 1 feature
    "atr_periods_multi": [7, 14, 21],  # 3 ATR variants
    
    # Historical Volatility - Realized vol at different timescales
    "hist_vol_periods": [10, 20, 60],  # 3 features (rolling std dev of log returns)
    
    # Keltner Channels - ATR-based envelope
    "keltner_params": {
        "period": 20,
        "atr_period": 14,
        "multiplier": 2,
    },  # 3 features: middle, upper, lower
    
    # Donchian Channels - High/Low breakout bands
    "donchian_period": 20,  # 2 features: upper, lower
    
    # ==================== VOLUME INDICATORS (~8 features) ====================
    # VWAP - Intraday benchmark (institutional execution target)
    "vwap": True,  # 1 feature
    
    # On-Balance Volume - Cumulative volume (trend confirmation)
    "obv": True,  # 1 feature
    
    # Volume Rate of Change - Unusual activity detection
    "volume_roc_periods": [5, 10, 20],  # 3 features
    
    # Accumulation/Distribution Line - Volume flow
    "ad_line": True,  # 1 feature
    
    # Volume Weighted Moving Average
    "vwma_period": 20,  # 1 feature
    
    # ==================== PATTERN RECOGNITION (~5+ features) ====================
    # Candlestick Patterns (binary/categorical features)
    "candlestick_patterns": {
        "enabled": True,
        "patterns": [
            "doji",           # Indecision
            "hammer",         # Bullish reversal
            "shooting_star",  # Bearish reversal
            "engulfing",      # Strong reversal
            "morning_star",   # Bullish reversal
            "evening_star",   # Bearish reversal
        ]
    },  # ~6 binary features
    
    # ==================== SUPPORT/RESISTANCE (~3 features) ====================
    # Support and Resistance Levels
    "support_resistance": {
        "enabled": True,
        "lookback": 20,
        "num_levels": 3,
    },  # Distance to nearest S/R levels
    
    # ==================== FIBONACCI LEVELS (~4 features) ====================
    # Fibonacci Retracement - Reversal zones
    "fibonacci": {
        "enabled": True,
        "lookback": 50,
        "levels": [0.236, 0.382, 0.500, 0.618],  # Key retracement levels
    },  # Distance to Fib levels
    
    # ==================== ADDITIONAL DERIVED FEATURES ====================
    # Price-based features
    "price_features": {
        "returns": [1, 5, 10, 20],  # Log returns at multiple periods
        "high_low_ratio": True,      # Daily range
        "close_position": True,      # Close position in daily range
    },
}

# Data Storage Paths (in-memory or file-based)
DATA_PATHS = {
    "raw": "data/raw",
    "validated": "data/validated",
    "features": "data/features",
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Timezone
TIMEZONE = "UTC"
