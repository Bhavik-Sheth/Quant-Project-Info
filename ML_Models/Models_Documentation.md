# ML Models Documentation

This directory contains various machine learning models for financial forecasting and analysis. The models are categorized into Volatility Forecasting, Direction Prediction, and Regime Classification.

## 1. Volatility Forecasting
**File:** `Volatility_Forecasting.py`

This module focuses on predicting the future volatility of financial assets. It employs a mix of classical statistical models and modern machine learning techniques.

### Classes

#### `Baseline_Pred`
- **Purpose**: Establishes a baseline for volatility prediction using linear regression.
- **Algorithm**: `LinearRegression` (Scikit-Learn).
- **Inputs**: DataFrame of features and target volatility series.

#### `Non_linear_Models`
- **Purpose**: Captures non-linear relationships in volatility data.
- **Algorithms**:
    - `RandomForestRegressor`: Ensemble learning method.
    - `XGBRegressor`: Gradient boosting framework.

#### `Volatility_Models`
- **Purpose**: Implements classical time-series volatility models.
- **Methods**:
    - `garch_train()`: Trains a standard GARCH(1,1) model to capture volatility clustering.
    - `egarch_train()`: Trains an EGARCH(1,1) model to handle asymmetric volatility shocks (leverage effect).
    - `predict_volatility(horizon)`: Returns the standard deviation forecast for the next `horizon` steps.

#### `LSTM_Volatility_Model`
- **Purpose**: Deep learning approach for volatility regression.
- **Algorithm**: Long Short-Term Memory (LSTM) Neural Network.
- **Architecture**:
    - Input Layer -> LSTM (128 units) -> Dropout -> LSTM (64 units) -> Dense (32) -> Output.
- **Key Features**:
    - Uses a lookback window (default 60 days).
    - Predicts realized volatility for a future horizon (default 5 days).

---

## 2. Direction Prediction
**File:** `direction_pred.py`

This module aims to predict the market direction (Up/Down) based on input features.

### Classes

#### `Baseline_Pred`
- **Purpose**: Baseline classification model.
- **Algorithm**: `LogisticRegression`.

#### `Extract_Rules`
- **Purpose**: Interpretable model to extract decision rules.
- **Algorithm**: `DecisionTreeClassifier`.
- **Key Method**: `get_rules()` returns JSON-formatted decision rules extracted from the tree structure.

#### `Feature_importance`
- **Purpose**: Identifies the most significant features driving market direction.
- **Algorithm**: `RandomForestClassifier`.
- **Key Method**: `rank_features(n_top)` returns the top N most important features.

#### `XGBoost_Pred`
- **Purpose**: High-performance gradient boosting for classification.
- **Algorithm**: `XGBClassifier`.

#### `FF_NN` (Feed Forward Neural Network)
- **Purpose**: Captures non-linear complex patterns.
- **Architecture**: Dense (128) -> Dense (64) -> Dense (32) -> Sigmoid Output.

#### `LSTM_Pred`
- **Purpose**: Captures temporal dependencies in sequential data.
- **Architecture**: LSTM (128) -> LSTM (64) -> Dense (1, Sigmoid).

---

## 3. Regime Classification
**File:** `Regime_Classificaiton.py`

This module classifies the market into different states or "regimes" (e.g., Range, Trend Up, Crisis).

### Classes

#### `Regime_Classifier`
- **Purpose**: Dual-model approach to detect market regimes.
- **Components**:
    1.  **Random Forest Classifier (`train_rf`)**:
        -   Analyzes the current day's snapshot to classify the regime.
        -   Uses `class_weight='balanced'` to handle rare events like crises.
    2.  **LSTM Classifier (`train_lstm`)**:
        -   Analyzes a sequence of past behavior (default 60 days) to classify based on transitions.
        -   Outputs probabilities for each regime class.
- **Key Method**: `predict_regime(current_sequence)` uses the LSTM model to predict the current market status labels.

---

## 4. Generative Models
**File:** `GAN.py`

*Currently empty/under development.*
