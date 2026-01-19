# Changelog

## [2026-01-19] - Enhanced main.py with Technical Indicator Display

### ✅ Added
- **Technical Indicator Display**: `main.py` now automatically displays calculated technical indicators after pipeline completion
  - Shows latest values for: Close Price, SMA (20/50/200), EMA (12/26), MACD, RSI, Bollinger Bands, ATR, ADX, VWAP, OBV, Stochastic
  - Includes status indicators (e.g., RSI: Overbought/Neutral/Oversold, ADX: Strong/Weak trend)
  - Displays total records and date range for each symbol/timeframe
- **Interactive Mode**: Built into `main.py` (no longer needs separate file)
- **CLI Arguments**: `main.py --symbols=AAPL,MSFT --timeframes=1h,1d`
- **Help Command**: `python3 main.py --help`

### 🔄 Changed
- Consolidated all interactive functionality into `main.py`
- Enhanced success message to show technical indicator values
- Updated `QUICKSTART.md` to reflect new usage

### ❌ Removed
- **run_interactive.py**: Deleted (functionality merged into `main.py`)

### 📊 Usage Examples

**Interactive Mode** (default):
```bash
python3 main.py
```

**CLI Mode**:
```bash
python3 main.py --symbols=AAPL --timeframes=1d
```

**Quick CLI**:
```bash
python3 quick_run.py AAPL MSFT
```

**Programmatic**:
```python
from main import run_full_pipeline

results = run_full_pipeline(
    symbols=['AAPL'],
    timeframes=['1d'],
    interactive=True  # Shows indicator summary
)
```

### 📈 Output Example

After running the pipeline, you'll see:

```
📈 TECHNICAL INDICATORS - AAPL (1d)
--------------------------------------------------------------------------------
Close Price:              $185.23
SMA 20:                  $182.45
SMA 50:                  $178.90
EMA 12:                  $184.12
MACD:                    1.2345
RSI (14):                65.43 (Neutral)
Bollinger Upper:         $190.23
Bollinger Lower:         $174.67
ATR (14):                3.45
ADX (14):                28.56 (Strong trend)
VWAP:                    $183.89
OBV:                     1,234,567,890

Total Records:           252
Date Range:              2025-01-20 to 2026-01-19
```

### 🎯 Benefits
- **Single Entry Point**: One file (`main.py`) for all use cases
- **Immediate Feedback**: See indicator values without opening CSV files
- **Better UX**: Clear, formatted output with status indicators
- **Simplified Project**: Fewer files, less confusion

