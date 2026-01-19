# Combined main.py - Enhanced Interactive Pipeline

## 🎯 What Changed

The `main.py` file now combines the best features of both `main.py` and `run_interactive.py` into a single, user-friendly script.

## ✨ New Features

### 1. **Interactive by Default**
When you run `python3 main.py` with no arguments, it automatically starts the interactive mode:
- ✅ Beautiful UI with emojis (📊, ⏰, 📈, ✅, 💡)
- ✅ Rich formatted prompts
- ✅ Popular symbols suggestions
- ✅ Timeframe descriptions (7 days, 60 days, etc.)
- ✅ Input validation (requires at least one symbol)
- ✅ Detailed configuration preview
- ✅ Success summary with file paths

### 2. **Enhanced User Experience**

**Before (old main.py)**:
```
Enter stock symbols (comma-separated)
Examples: AAPL, MSFT,GOOGL,AMZN or ^GSPC,^DJI for indices
Press Enter to use defaults (AAPL, MSFT, GOOGL, AMZN, TSLA)

Stock symbols: 
```

**After (new main.py)**:
```
📊 ENTER STOCK SYMBOLS
--------------------------------------------------------------------------------
Enter one or more stock symbols (comma-separated)

Examples:
  • Single stock:    AAPL
  • Multiple stocks: AAPL, MSFT, GOOGL
  • With indices:    AAPL, ^GSPC, ^DJI

Popular symbols:
  Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX
  Indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)

👉 Enter symbols: 
```

### 3. **Better Configuration Display**

Shows exactly what will be processed:
```
================================================================================
                         PIPELINE CONFIGURATION
================================================================================

📈 Symbols (2):
   1. AAPL
   2. TSLA

⏰ Timeframes (2):
   • 1h  - 2 years
   • 1d  - 1 year

================================================================================

✅ Start pipeline? (Y/n): 
```

### 4. **Enhanced Success Summary**

After completion, shows all generated files:
```
================================================================================
                              ✅ SUCCESS!
================================================================================

📁 Generated Files:
--------------------------------------------------------------------------------

AAPL (1h):
  • data/raw/1h/AAPL_1h_raw.csv
  • data/validated/1h/clean/AAPL_1h_clean.csv
  • data/features/1h/AAPL_1h_features.csv

AAPL (1d):
  • data/raw/1d/AAPL_1d_raw.csv
  • data/validated/1d/clean/AAPL_1d_clean.csv
  • data/features/1d/AAPL_1d_features.csv

================================================================================

💡 Next Steps:
  1. Check validation log: data/validated/validation_log.csv
  2. Load features: import pandas as pd; df = pd.read_csv('data/features/...')
  3. Analyze your data with the technical indicators!

================================================================================
```

## 📖 Usage Examples

### 1. **Interactive Mode (Default)**
```bash
python3 main.py
```
Just run it! The script will:
1. Ask you which stocks you want
2. Ask which timeframes
3. Show you what will be processed
4. Confirm before running
5. Process the data
6. Show you exactly what files were created

### 2. **Command-Line Symbols**
```bash
python3 main.py AAPL TSLA META
```
Quick mode - processes specified symbols with all timeframes

### 3. **Explicit Interactive**
```bash
python3 main.py -i
# or
python3 main.py --interactive
```

### 4. **Help**
```bash
python3 main.py --help
```
Shows all usage options

### 5. **Programmatic (Python Code)**
```python
from main import run_full_pipeline

# Process specific stocks
results = run_full_pipeline(['AAPL', 'MSFT'], ['1h', '1d'])

# Interactive mode from Python
results = run_full_pipeline(interactive=True)
```

## 🔄 What Happens When You Run It

### Flow Diagram:
```
1. Run: python3 main.py
   ↓
2. Welcome Screen (🚀 header)
   ↓
3. Enter Stock Symbols (📊 with examples)
   ↓
4. Select Timeframes (⏰ with descriptions)
   ↓
5. Review Configuration (📈 numbered list)
   ↓
6. Confirm (✅ Y/n prompt)
   ↓
7. Stage 1: Data Ingestion (Yahoo Finance API)
   ↓
8. Stage 2: Data Validation (Quality checks)
   ↓
9. Stage 3: Feature Engineering (30+ indicators)
   ↓
10. Success Summary (📁 file list + 💡 next steps)
```

## 🎁 Benefits

1. **User-Friendly**: No need to remember syntax - the script guides you
2. **Validation**: Can't proceed with invalid input - prevents errors
3. **Transparency**: See exactly what will happen before it runs
4. **Feedback**: Clear success messages show what was created
5. **Flexibility**: Still supports all old modes (CLI, programmatic)
6. **One File**: No need to choose between main.py and run_interactive.py

## 🚀 Quick Start

**Absolute beginner? Just run this:**
```bash
python3 main.py
```

Then enter:
- A stock you're interested in (e.g., `AAPL`)
- A timeframe (e.g., `1d` or just press Enter for all)
- Press `y` to confirm

The pipeline will:
✅ Fetch the data from Yahoo Finance  
✅ Clean and validate it  
✅ Calculate 30+ technical indicators  
✅ Save everything to CSV files  
✅ Tell you exactly where to find your data  

That's it! 🎉

## 📝 Migration Notes

### Old Workflow:
```bash
# Had to choose which script to use
python3 run_interactive.py  # For interactive
python3 main.py AAPL       # For CLI
```

### New Workflow:
```bash
# Just use main.py for everything
python3 main.py            # Interactive (auto-detected)
python3 main.py AAPL       # CLI still works
```

### Backwards Compatibility:
✅ All old functionality preserved  
✅ Programmatic imports still work  
✅ CLI arguments still work  
✅ Can still use run_interactive.py if preferred  

## 🎯 Summary

**The new main.py is:**
- 🎨 More beautiful (emojis, formatting)
- 🛡️ More robust (input validation)
- 📚 More informative (better examples, clear output)
- 🚀 Easier to use (interactive by default)
- 🔧 Still powerful (all old features preserved)

**Perfect for:**
- First-time users learning the pipeline
- Daily interactive use
- Teaching/demonstrating
- Quick ad-hoc analysis

**Your workflow is now simple:**
```bash
python3 main.py
# Enter stock name → Clean → Validate → Feature Engineering → Done! ✅
```
