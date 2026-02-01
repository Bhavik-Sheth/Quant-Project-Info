# Quick Training Guide
## One-Time Model Training for CF-AI-SDE

---

## 📋 Pre-Training Checklist

```bash
cd backend

# 1. Verify Python environment
python --version  # Should be 3.8+

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Check config file exists
ls config.yaml

# 4. (Optional) Start MongoDB
# If not available, TinyDB will be used automatically
```

---

## 🚀 Run Training (One Command)

```bash
python train_production_models.py --full-pipeline
```

**This will automatically:**
1. ✅ Fetch 26 symbols (AAPL, MSFT, GOOGL, etc.) - 5 years daily data
2. ✅ Engineer 70+ technical indicators (RSI, MACD, BB, etc.)
3. ✅ Validate data quality
4. ✅ Prepare train/val/test splits (80/10/10)
5. ✅ Train direction prediction models (XGBoost)
6. ✅ Train regime classification models (Random Forest)
7. ✅ Save models to MongoDB + disk backup
8. ✅ Generate training report

**Expected Time:** 45-60 minutes

---

## 📊 What Gets Trained

### Data Volume
- **26 symbols** across 5 sectors + indices
- **~1,260 trading days** (5 years: 2020-2025)
- **~32,760 samples** total
- **70+ features** per sample

### Models Trained

**1. Direction Predictor (XGBoost)**
- Predicts next-day price direction (Up/Down/Neutral)
- Target accuracy: > 55% (baseline: 50%)
- File: `data/models/direction_xgboost_v1.0.pkl`

**2. Regime Classifier (Random Forest)**
- Identifies market regime (Bull/Bear/Sideways/High Vol)
- Target accuracy: > 70%
- File: `data/models/regime_rf_v1.0.pkl`

**3. Volatility Baseline**
- Rolling standard deviation (20-day window)
- Fallback for volatility predictions

---

## 📍 Output Locations

### Models
```
data/models/
├── direction_xgboost_v1.0.pkl    # Direction predictor
└── regime_rf_v1.0.pkl            # Regime classifier
```

### Data
```
data/
├── raw/              # Raw OHLCV data (CSV)
├── features/         # Engineered features (CSV)
└── fallback/         # TinyDB backup (if MongoDB unavailable)
```

### Reports
```
training_pipeline.log      # Detailed log file
training_report.json       # Performance metrics
```

---

## 🔍 Verify Training Success

### Check Log File
```bash
tail -n 50 training_pipeline.log
```

Look for:
```
✅ Direction XGBoost - Train Acc: 0.XXX, Test Acc: 0.XXX
✅ Regime Random Forest - Train Acc: 0.XXX, Test Acc: 0.XXX
🎉 PIPELINE COMPLETE!
```

### Check Models Exist
```bash
ls -lh data/models/
```

Should see:
```
direction_xgboost_v1.0.pkl    ~500KB
regime_rf_v1.0.pkl            ~1MB
```

### Check Report
```bash
cat training_report.json | head -n 30
```

Should contain:
```json
{
  "ingestion": {
    "status": "success",
    "records": 32760
  },
  "training": {
    "direction": {
      "xgboost": {
        "test_acc": 0.573
      }
    },
    "regime": {
      "random_forest": {
        "test_acc": 0.742
      }
    }
  }
}
```

---

## 🧪 Test Trained Models

### Option 1: Python Script
```python
from logical_pipe import TradingSystemAPI, ConfigLoader

# Initialize API
api = TradingSystemAPI(ConfigLoader("config.yaml"))

# Test direction prediction
result = api.generate_signals(
    symbol="AAPL",
    strategy="ml_enhanced",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
print(result)
```

### Option 2: Via API
```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "strategy": "ml_enhanced",
    "lookback_period": 90
  }'
```

Expected response:
```json
{
  "status": "success",
  "signals": [
    {"date": "2024-12-01", "signal": "buy", "confidence": 0.73},
    {"date": "2024-12-02", "signal": "hold", "confidence": 0.52}
  ],
  "strategy": "ml_enhanced",
  "count": 30
}
```

---

## ⚠️ Troubleshooting

### Issue: "No module named 'Data_inges_fe'"
**Fix:** Import path issue
```bash
cd backend
python train_production_models.py --full-pipeline
```

### Issue: MongoDB connection error
**Fix:** Automatic TinyDB fallback
- Training will continue using TinyDB
- Models saved to `data/fallback/ml_models.json`
- No action needed!

### Issue: Memory error during training
**Fix:** Reduce data volume
Edit `train_production_models.py`:
```python
# Line 67-70: Reduce symbols
TRAINING_SYMBOLS = {
    "indices": ["^GSPC", "^VIX"],  # Keep only 2 indices
    "tech": ["AAPL", "MSFT"],      # Keep only 2 tech stocks
    ...
}

# Line 81: Reduce lookback period
PRIMARY_LOOKBACK_YEARS = 2  # Instead of 5
```

### Issue: Training too slow
**Expected times:**
- Data ingestion: 10-15 minutes
- Feature engineering: 5 minutes
- Model training: 20-30 minutes
- **Total: 45-60 minutes**

If significantly slower:
- Check internet connection (Yahoo Finance API)
- Consider using fewer symbols
- Use GPU for LSTM training (if available)

---

## 📈 Expected Performance

### Minimum Acceptable Metrics

**Direction Predictor:**
- Test Accuracy: > 55% ✅ (Random guess: 50%)
- Precision: > 0.55
- Recall: > 0.50

**Regime Classifier:**
- Test Accuracy: > 70% ✅
- Macro F1-score: > 0.68

### Good Performance

**Direction Predictor:**
- Test Accuracy: 57-62% 🎯
- Sharpe Ratio (backtested): > 1.0

**Regime Classifier:**
- Test Accuracy: 72-78% 🎯
- Correct regime identification > 75% of time

---

## 🔄 Retraining (Future)

To retrain with new data:

```bash
# Update date range in script (lines 84-85)
# START_DATE = datetime(2021, 1, 1)  # New start date
# END_DATE = datetime.now()          # Today

# Run training again
python train_production_models.py --full-pipeline

# Models will be saved with new version
# direction_xgboost_v2.0.pkl
```

**Recommended:** Retrain quarterly or when:
- Market regime changes significantly
- Model accuracy drops > 5% on live data
- New symbols added to trading universe

---

## ✅ Post-Training Checklist

- [ ] Log file shows "🎉 PIPELINE COMPLETE!"
- [ ] `data/models/` contains .pkl files
- [ ] `training_report.json` exists
- [ ] Test accuracy > 55% for direction
- [ ] Test accuracy > 70% for regime
- [ ] API endpoint returns ml_enhanced signals
- [ ] No errors in log file

---

## 🎯 Next Steps After Training

1. **Test API endpoints** with trained models
2. **Run backtest** on historical data
3. **Validate performance** against benchmark
4. **Deploy to production** (if metrics acceptable)
5. **Set up monitoring** for live performance

---

## 📞 Quick Commands Reference

```bash
# Full training pipeline
python train_production_models.py --full-pipeline

# Check training log
tail -f training_pipeline.log

# Verify models
ls -lh data/models/

# Test models
python -c "from logical_pipe import TradingSystemAPI, ConfigLoader; api = TradingSystemAPI(ConfigLoader('config.yaml')); print('Models loaded:', api.model_manager.models)"

# Start API with trained models
uvicorn api.main:app --reload --port 8000
```

---

**Ready to train?** Run: `python train_production_models.py --full-pipeline`

**Questions?** Check `documentation/TRAINING_PLAN.md` for detailed explanation.
