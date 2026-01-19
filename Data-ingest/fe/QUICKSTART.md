# 🚀 Quick Start Guide

## Three Ways to Run the Pipeline

### 1️⃣ Interactive Mode (Recommended for Beginners)

**Easiest way** - The script will prompt you for input:

```bash
python3 main.py
```

**Example session:**
```
👉 Enter symbols: AAPL, MSFT, NVDA
👉 Timeframes (or Enter for all): 1h,1d
👉 Proceed? (Y/n): y
```

**New Feature**: Now automatically displays technical indicator values at the end!

---

### 2️⃣ Command Line Mode (Quick & Simple)

**Fast way** - Specify symbols directly:

```bash
# Single stock
python3 quick_run.py AAPL

# Multiple stocks
python3 quick_run.py AAPL MSFT GOOGL

# With specific timeframes
python3 quick_run.py NVDA --timeframes 1d

# Multiple stocks and timeframes
python3 quick_run.py AAPL MSFT TSLA --timeframes 1h,1d

# Indices
python3 quick_run.py "^GSPC" "^DJI" --timeframes 1d
```

---

### 3️⃣ Python Script Mode (Advanced)

**Programmatic way** - Use in your own scripts:

```python
from main import run_full_pipeline

# Run for specific stocks
results = run_full_pipeline(
    symbols=["AAPL", "MSFT", "NVDA"],
    timeframes=["1h", "1d"]
)

# Access the data
features = results['features']['1d']['AAPL']
```

---

## 📊 Popular Stock Symbols

### Tech Stocks
- `AAPL` - Apple
- `MSFT` - Microsoft
- `GOOGL` - Google/Alphabet
- `AMZN` - Amazon
- `NVDA` - NVIDIA
- `META` - Meta/Facebook
- `TSLA` - Tesla
- `NFLX` - Netflix

### Indices
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones
- `^IXIC` - NASDAQ

### Other Popular
- `JPM` - JPMorgan Chase
- `V` - Visa
- `WMT` - Walmart
- `DIS` - Disney

---

## ⏰ Timeframe Options

| Code | Description | History Available |
|------|-------------|-------------------|
| `1m` | 1 minute    | Last 7 days       |
| `5m` | 5 minutes   | Last 60 days      |
| `1h` | 1 hour      | Last 2 years      |
| `1d` | 1 day       | Last 1 year       |

---

## 📁 Output Files

After running the pipeline, you'll find:

```
data/
├── raw/
│   └── {timeframe}/
│       └── {SYMBOL}_{timeframe}_raw.csv
│
├── validated/
│   └── {timeframe}/
│       └── clean/
│           └── {SYMBOL}_{timeframe}_clean.csv
│
└── features/
    └── {timeframe}/
        └── {SYMBOL}_{timeframe}_features.csv  ← Use this!
```

**The features file contains 44 columns including:**
- OHLCV data
- 30+ technical indicators (SMA, EMA, RSI, MACD, etc.)

---

## 💡 Quick Examples

### Example 1: Analyze Apple Stock (Daily)
```bash
python3 quick_run.py AAPL --timeframes 1d
```

### Example 2: Compare Tech Giants (Hourly + Daily)
```bash
python3 quick_run.py AAPL MSFT GOOGL NVDA --timeframes 1h,1d
```

### Example 3: S&P 500 Analysis
```bash
python3 quick_run.py "^GSPC" --timeframes 1d
```

### Example 4: Interactive for Custom Selection
```bash
python3 run_interactive.py
# Then enter: TSLA, NVDA
# Then enter: 1h,1d
```

---

## 📊 Load and Analyze Data

After running the pipeline:

```python
import pandas as pd

# Load feature data
df = pd.read_csv('data/features/1d/AAPL_1d_features.csv')

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check available features
print(df.columns.tolist())

# View latest data with indicators
print(df[['timestamp', 'close', 'sma_20', 'rsi_14', 'macd']].tail())

# Get most recent RSI
latest_rsi = df['rsi_14'].iloc[-1]
print(f"Current RSI: {latest_rsi:.2f}")
```

---

## ⚡ Tips

1. **Start Small**: Try one stock with daily data first
   ```bash
   python3 quick_run.py AAPL --timeframes 1d
   ```

2. **Use Interactive Mode**: When learning or testing
   ```bash
   python3 run_interactive.py
   ```

3. **Intraday Data**: Limited history (7-60 days)
   - `1m` data only goes back 7 days
   - `5m` data only goes back 60 days

4. **Check Validation Log**: After running
   ```bash
   cat data/validated/validation_log.csv
   ```

5. **Multiple Runs**: Each run overwrites previous data for same symbols

---

## 🆘 Troubleshooting

### Symbol Not Found
- Make sure symbol is valid on Yahoo Finance
- Use correct format: `AAPL` not `Apple`
- For indices, use `^` prefix: `^GSPC`

### No Data Returned
- Check if market was open during requested period
- Some symbols may have limited historical data
- Try a different timeframe

### Too Slow
- Reduce number of symbols
- Use fewer timeframes
- Start with daily data (`1d`) only

---

## 🎯 What You Get

For each stock:
- ✅ Raw OHLCV data
- ✅ Validated & cleaned data
- ✅ 30+ technical indicators
- ✅ Validation issue log
- ✅ All in CSV format

All with **zero look-ahead bias** and **UTC timestamps**!

---

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Technical Details**: See `TECHNICAL_DOCS.md`
- **Usage Examples**: Run `python3 examples.py`

---

**Happy analyzing! 📈**
