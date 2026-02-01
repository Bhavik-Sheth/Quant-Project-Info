# Backtesting & Risk Management Framework

**Version 2.0.0** - Production Ready  
**Status:** 95% Complete

A comprehensive backtesting system for AI-driven trading strategies with institutional-grade features.

## 🎯 Quick Start

```python
from Backtesting_risk import BacktestEngine, TradeDecision, Action

engine = BacktestEngine(initial_capital=100000)
metrics = engine.run(market_data, features, decisions)

print(f"Return: {metrics.total_return:.2%}")
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")

engine.export_all_results('results/')
```

## ✨ Features

- ✅ **Realistic Execution**: T+1 delays, slippage, transaction costs
- ✅ **Multi-Layer Risk Management**: Position limits, volatility control, circuit breakers  
- ✅ **Complete Audit Trail**: Model predictions, SHAP values, full logging
- ✅ **20+ Performance Metrics**: Returns, risk, risk-adjusted, trade stats
- ✅ **Walk-Forward Validation**: Out-of-sample testing
- ✅ **Export Everything**: CSV exports for all data

## 📁 File Structure (7 Files Total)

```
Backtesting_risk/
├── 📦 CORE (4 files)
│   ├── models.py        # Data structures
│   ├── backtesting.py   # All engines
│   ├── analysis.py      # Performance metrics
│   └── __init__.py      # Package exports
│
├── 📝 DOCS (2 files)
│   ├── README.md         # This file
│   └── DOCUMENTATION.md  # Complete reference
│
└── 💡 EXAMPLES (1 file)
    └── examples.py       # Usage examples
```

**Reduced from 25+ files to 7! (72% reduction)**

## 🚀 Core Components

### 1. Execution Simulation
- T+1 execution delays
- Slippage modeling (0.1%)
- Transaction costs

### 2. Risk Management
- Position limits (10% max per position)
- Sector exposure (30% max per sector)
- Volatility control (20% max)
- Circuit breaker (20% drawdown)

### 3. Performance Metrics
20+ metrics including:
- Returns, Sharpe, Sortino, Calmar
- Max drawdown, volatility, VaR
- Win rate, profit factor

### 4. Audit Trail
- Market data snapshots
- Model predictions
- SHAP values
- Risk events
- Full context

## 📊 Requirements Status

| Requirement | Status |
|------------|--------|
| 8.1 Execution Simulation | ✅ 100% |
| 8.2 Risk Management | ✅ 100% |
| 8.3 Performance Metrics | ✅ 100% |
| 10.1 Position Limits | ✅ 100% |
| 10.2 Volatility Control | ✅ 100% |
| 10.3 Circuit Breaker | ✅ 100% |
| 10.4 Audit Trail | ✅ 100% |

## 📖 Documentation

- **README.md** - This file (quick start)
- **DOCUMENTATION.md** - Complete API reference with examples
- **examples.py** - Working code examples

## 🔧 API

```python
from Backtesting_risk import (
    BacktestEngine,
    ExecutionConfig,
    RiskConfig,
    TradeDecision,
    Action,
    PerformanceMetrics
)
```

## 🎓 Next Steps

1. See `DOCUMENTATION.md` for complete API reference
2. Run `examples.py` for working examples
3. Integrate your ML models and AI agents
4. Run backtests and analyze results

---

**Version:** 2.0.0  
**Last Updated:** February 1, 2026
