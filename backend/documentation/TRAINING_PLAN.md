# One-Time Model Training Plan
## CF-AI-SDE Trading System

---

## 🎯 Objective
Train generalized, production-ready ML models using diverse market data across multiple sectors, timeframes, and market conditions.

---

## 📊 Data Ingestion Strategy

### **1. Symbol Selection (Diversified Portfolio)**

#### **Major Indices (6 symbols)**
- `^GSPC` - S&P 500 (US Large Cap benchmark)
- `^IXIC` - NASDAQ (Tech-heavy index)
- `^DJI` - Dow Jones (Blue-chip companies)
- `^NSEI` - NIFTY 50 (India market)
- `^NSEBANK` - BANKNIFTY (India banking sector)
- `^VIX` - Volatility Index (Fear gauge)

#### **Liquid Equities (20 stocks across 5 sectors)**

**Technology (5):**
- AAPL, MSFT, GOOGL, NVDA, META

**Financials (4):**
- JPM, BAC, GS, V

**Healthcare (3):**
- JNJ, UNH, PFE

**Consumer (4):**
- WMT, HD, MCD, NKE

**Energy (4):**
- XOM, CVX, COP, SLB

**Total: 26 symbols** (6 indices + 20 equities)

---

### **2. Timeframe Selection**

**Primary Timeframe:**
- `1d` (Daily) - **Main training data**
  - Lookback: **5 years** (2020-01-01 to 2025-02-01)
  - Rationale: Captures multiple market regimes (COVID crash, bull market, rate hikes)
  - Volume: ~1,260 trading days × 26 symbols = **32,760 samples**

**Secondary Timeframe (Optional):**
- `1h` (Hourly) - For intraday pattern learning
  - Lookback: **6 months** (2024-08-01 to 2025-02-01)
  - Rationale: Recent price action, less storage
  - Volume: ~120 days × 6.5 hours × 26 symbols = **20,280 samples**

---

### **3. Feature Engineering**

Using existing `Data-inges-fe/src/features/technical_indicators.py`:

**Category 1: Trend Indicators (15)**
- MA (5, 10, 20, 50, 200), EMA (12, 26), MACD, ADX, CCI, etc.

**Category 2: Momentum Indicators (15)**
- RSI (7, 14, 21), Stochastic, Williams %R, ROC, MFI, etc.

**Category 3: Volatility Indicators (10)**
- Bollinger Bands, ATR, Keltner Channels, Donchian, Standard Dev

**Category 4: Volume Indicators (10)**
- OBV, Volume Ratio, VWAP, Accumulation/Distribution, CMF

**Category 5: Pattern Recognition (10)**
- Support/Resistance, Breakouts, Divergence, Higher Highs/Lows

**Category 6: Statistical Features (15)**
- Returns (1d, 5d, 20d), Z-score, Autocorrelation, Entropy

**Total: 70+ features per sample**

---

### **4. Data Quality Requirements**

✅ **Validation Checks** (from `Data-inges-fe/src/validation/ohlcv_checks.py`):
1. **Completeness**: < 5% missing data per symbol
2. **OHLC Logic**: High ≥ Low, Open/Close within [Low, High]
3. **Price Sanity**: No > 50% daily moves (split detection)
4. **Volume Sanity**: > 0 for trading days
5. **Temporal Integrity**: No duplicate/missing timestamps
6. **No Look-Ahead Bias**: Features use only past data

---

## 🤖 Model Training Strategy

### **Model 1: Direction Predictor (XGBoost + LSTM)**

**Purpose:** Predict next-day price direction (Up/Down/Neutral)

**Training Data:**
- Features: 70 technical indicators
- Target: `sign(close_t+1 - close_t)`
  - `1` = Up (> +0.1%)
  - `0` = Neutral ([-0.1%, +0.1%])
  - `-1` = Down (< -0.1%)
- Train: 2020-2023 (80%)
- Validation: 2024 H1 (10%)
- Test: 2024 H2 - 2025 (10%)

**Models:**
1. **XGBoost Classifier**
   - Tree-based, handles non-linearity
   - Feature importance extraction
   - Fast inference

2. **LSTM Network**
   - Sequence length: 20 days
   - Captures temporal dependencies
   - Complement to XGBoost

**Success Metric:** Accuracy > 55% (baseline: 50%)

---

### **Model 2: Volatility Forecaster (GARCH + LSTM)**

**Purpose:** Predict next-day realized volatility

**Training Data:**
- Features: Historical volatility, ATR, Bollinger width, Volume
- Target: 20-day rolling standard deviation of returns
- Same train/val/test split as above

**Models:**
1. **GARCH(1,1)**
   - Specialized for volatility clustering
   - Captures mean reversion

2. **LSTM Network**
   - Learns volatility regime shifts
   - Non-parametric approach

**Success Metric:** RMSE < naive forecast (using yesterday's vol)

---

### **Model 3: Regime Classifier (Random Forest)**

**Purpose:** Identify market regime (Bull/Bear/Sideways/High Vol)

**Training Data:**
- Features: 
  - 50/200 MA crossover
  - VIX level
  - Trend strength (ADX)
  - Breadth indicators
- Target: Regime labels based on:
  - Bull: 50MA > 200MA, VIX < 20, Positive returns
  - Bear: 50MA < 200MA, VIX > 30, Negative returns
  - Sideways: Choppy, low ADX
  - High Vol: VIX > 30, large swings

**Model:** Random Forest (100 trees)

**Success Metric:** Classification accuracy > 70%

---

## 🔧 Implementation Plan

### **Phase 1: Data Ingestion (10-15 minutes)**

```bash
# Run ingestion pipeline
cd backend/Data-inges-fe
python main.py
```

**Expected Output:**
- `data/raw/` - Raw OHLCV data (CSV files)
- `data/features/` - Engineered features (CSV files)
- `data/validated/` - Cleaned data passing all checks

**Storage:**
- MongoDB: `trading_system.market_data` collection
- TinyDB Fallback: `data/fallback/market_data.json`

---

### **Phase 2: Feature Engineering (5 minutes)**

Already integrated in Phase 1 via:
- `Data-inges-fe/src/features/technical_indicators.py`
- Automatically computes 70+ indicators

**Verify:**
```python
import pandas as pd
df = pd.read_csv('data/features/AAPL_1d_features.csv')
print(df.columns)  # Should see RSI, MACD, BB, etc.
```

---

### **Phase 3: Model Training (20-30 minutes)**

**Script:** `train_production_models.py` (to be created)

```python
# High-level flow:
1. Load all ingested data from MongoDB/TinyDB
2. Prepare training sets (train/val/test splits)
3. Train direction model (XGBoost + LSTM)
4. Train volatility model (GARCH + LSTM)
5. Train regime model (Random Forest)
6. Evaluate on test set
7. Save models to MongoDB + disk
8. Log performance metrics
```

**Expected Training Time:**
- Direction XGBoost: 2-3 minutes
- Direction LSTM: 5-8 minutes
- Volatility GARCH: 1 minute
- Volatility LSTM: 5-8 minutes
- Regime RF: 1-2 minutes

**Total: ~25 minutes on CPU, ~10 minutes on GPU**

---

### **Phase 4: Model Validation & Storage (5 minutes)**

**Checks:**
1. ✅ Models saved to MongoDB: `trading_system.ml_models` collection
2. ✅ Fallback to disk: `data/models/` directory
3. ✅ Performance metrics logged
4. ✅ Test set accuracy meets thresholds

**Verify:**
```python
from logical_pipe import TradingSystemAPI, ConfigLoader
api = TradingSystemAPI(ConfigLoader("config.yaml"))
result = api.model_manager.load_model("direction", version="1.0")
print(f"Model loaded: {result is not None}")
```

---

## 📋 Implementation Checklist

### **Prerequisites:**
- [ ] MongoDB running (or accept TinyDB fallback)
- [ ] `config.yaml` configured
- [ ] All dependencies installed (`pip install -r requirements.txt`)

### **Execution Steps:**

**Step 1: Create Training Script**
- [ ] Create `train_production_models.py` in backend/
- [ ] Implement data loading from existing pipeline
- [ ] Implement train/val/test splits
- [ ] Implement model training loops
- [ ] Implement model persistence

**Step 2: Run Data Ingestion**
```bash
cd backend
python train_production_models.py --stage ingest
```

**Step 3: Train Models**
```bash
python train_production_models.py --stage train
```

**Step 4: Validate Models**
```bash
python train_production_models.py --stage validate
```

**Step 5: Deploy Models**
```bash
python train_production_models.py --stage deploy
```

---

## 📈 Expected Outcomes

### **Data Volume:**
- **26 symbols** × **1,260 days** = **32,760 samples** (daily)
- After feature engineering: **32,760 rows** × **70+ columns**
- Storage: ~50-100 MB in MongoDB/TinyDB

### **Model Performance (Target):**
- Direction Accuracy: **55-60%** (vs 50% baseline)
- Volatility RMSE: **< 0.015** (vs 0.020 naive)
- Regime Accuracy: **70-75%**

### **Model Artifacts:**
```
data/models/
├── direction_xgboost_v1.0.pkl
├── direction_lstm_v1.0.h5
├── volatility_garch_v1.0.pkl
├── volatility_lstm_v1.0.h5
└── regime_rf_v1.0.pkl
```

### **Metadata:**
```json
{
  "model_name": "direction_xgboost",
  "version": "1.0",
  "trained_date": "2025-02-02",
  "training_samples": 26208,
  "test_accuracy": 0.573,
  "symbols": ["AAPL", "MSFT", ...],
  "feature_count": 72,
  "timeframe": "1d"
}
```

---

## ⚡ Quick Start Command

```bash
# One-command training (to be implemented)
cd backend
python train_production_models.py --full-pipeline
```

This will:
1. ✅ Ingest 26 symbols (5 years daily)
2. ✅ Engineer 70+ features
3. ✅ Train 5 models (XGBoost, LSTM, GARCH, LSTM, RF)
4. ✅ Validate on test set
5. ✅ Save to MongoDB + disk
6. ✅ Log performance report

**Estimated Time:** 45 minutes end-to-end

---

## 🚀 Next Steps

1. **Review this plan** - Confirm symbol selection and timeframes
2. **Create training script** - Implement `train_production_models.py`
3. **Run ingestion** - Fetch data using existing pipeline
4. **Train models** - Execute training with proper splits
5. **Integrate with API** - Enable model loading in `/signals/generate`

---

## ⚠️ Important Notes

1. **One-time training:** This is initial training. Retrain monthly/quarterly with new data.
2. **Generalization:** Using diverse symbols across sectors ensures models work on new stocks.
3. **No overfitting:** Strict train/val/test split prevents data leakage.
4. **Fallback ready:** TinyDB ensures training works even without MongoDB.
5. **Reproducible:** Fixed random seeds for consistent results.

---

**Ready to implement?** Let me know and I'll create the `train_production_models.py` script! 🎯
