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

# Default Symbols to Fetch
DEFAULT_EQUITY_SYMBOLS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "GOOGL", # Alphabet Inc.
    "AMZN",  # Amazon.com Inc.
    "TSLA",  # Tesla Inc.
]

DEFAULT_INDEX_SYMBOLS = [
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones Industrial Average
    "^IXIC",  # NASDAQ Composite
]

# Date Range Configuration
# Default: Last 30 days for intraday, 1 year for daily
DEFAULT_LOOKBACK_DAYS = {
    "1m": 7,      # Yahoo Finance limits 1m data to 7 days
    "5m": 60,     # Yahoo Finance limits 5m data to 60 days
    "1h": 730,    # 2 years
    "1d": 365,    # 1 year
}

# Validation Thresholds
VALIDATION_CONFIG = {
    "price_outlier_threshold": 0.20,  # 20% price change threshold
    "max_gap_multiplier": 2.0,        # Max gap = 2x expected interval
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    # Trend Indicators
    "sma_periods": [10, 20, 50, 200],
    "ema_periods": [12, 26],
    "macd_params": {
        "fast": 12,
        "slow": 26,
        "signal": 9,
    },
    "adx_period": 14,
    
    # Momentum Indicators
    "rsi_period": 14,
    "roc_periods": [5, 10, 20],
    "stoch_params": {
        "k_period": 14,
        "d_period": 3,
    },
    
    # Volatility Indicators
    "atr_period": 14,
    "bollinger_params": {
        "period": 20,
        "std_dev": 2,
    },
    "hist_vol_periods": [10, 20, 60],
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
