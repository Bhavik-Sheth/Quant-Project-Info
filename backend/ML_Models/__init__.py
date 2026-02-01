"""
Machine Learning Models Module

This module contains various ML models for quantitative trading:
- Direction Prediction (XGBoost, LSTM)
- Volatility Forecasting (GARCH, LSTM)
- Regime Classification
- Market GAN for synthetic data generation
"""

from .direction_pred import (
    XGBoost_Pred,
    LSTM_Pred,
    Feature_importance,
    Extract_Rules,
)

try:
    from .Volatility_Forecasting import (
        Volatility_Models,
        LSTM_Volatility_Model,
        VolatilityGARCH,
        VolatilityLSTM,
    )
except ImportError:
    pass

try:
    from .Regime_Classificaiton import Regime_Classifier, RegimeClassifier
except ImportError:
    pass

try:
    from .GAN import MarketGAN
except ImportError:
    pass

__all__ = [
    "XGBoost_Pred",
    "LSTM_Pred",
    "Feature_importance",
    "Extract_Rules",
    "Volatility_Models",
    "LSTM_Volatility_Model",
    "VolatilityGARCH",
    "VolatilityLSTM",
    "Regime_Classifier",
    "RegimeClassifier",
    "MarketGAN",
]
