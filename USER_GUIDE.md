# CF-AI-SDE Trading Platform - User Guide

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Platform Features](#platform-features)
4. [Backend-Frontend Mapping](#backend-frontend-mapping)
5. [Page-by-Page Guide](#page-by-page-guide)
6. [API Endpoints Reference](#api-endpoints-reference)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

CF-AI-SDE is an AI-powered multi-agent trading system that combines:
- **Machine Learning Models** - Direction prediction, volatility forecasting, regime classification
- **AI Agents** - 7 specialized agents for market analysis
- **Backtesting Engine** - Realistic execution simulation with risk management
- **RAG Mentor** - Trading knowledge Q&A system
- **Real-time Data** - Yahoo Finance integration with 70+ technical indicators

---

## 🚀 Getting Started

### Prerequisites
- Backend: Python 3.13+, MongoDB (optional)
- Frontend: Node.js 16+, npm

### Starting the Platform

#### 1. Start Backend Server
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### 2. Start Frontend Server
```bash
cd ui
npm run dev
```

Frontend will be available at:
- Application: http://localhost:3000

### Initial Setup
1. Check the **API Status Indicator** (bottom-right corner) shows "Backend Connected" (green)
2. Visit the home page to see the system overview
3. Navigate using the sidebar to access different features

---

## 🎨 Platform Features

### 1. **Market Data Dashboard** (Coming Soon)
- Real-time market data visualization
- Live price charts
- Volume analysis

### 2. **Strategy Builder** (`/strategy`)
- Visual strategy designer
- Define entry/exit rules
- View strategy JSON output

### 3. **Backtesting** (`/backtest`)
- Test strategies on historical data
- View performance metrics
- Analyze equity curves

### 4. **AI Mentor** (`/mentor`)
- Ask trading-related questions
- Get AI-powered insights
- Access trading knowledge base

### 5. **Technical Indicators** (`/indicators`)
- Browse 70+ technical indicators
- Learn indicator calculations
- View usage examples

---

## 🔗 Backend-Frontend Mapping

### Complete Architecture Map

| Frontend Section | Frontend Route | Backend Endpoint(s) | Backend Router | Purpose |
|-----------------|----------------|-------------------|----------------|---------|
| **Home Page** | `/` | `GET /health` | `health.py` | System status |
| **Market Data** | `/market` | `POST /data/ingest`<br>`GET /data/latest/{symbol}` | `data.py` | Data ingestion & retrieval |
| **Strategy Builder** | `/strategy` | `GET /signals/strategies`<br>`POST /signals/generate` | `signals.py` | Strategy management & signal generation |
| **Backtest** | `/backtest` | `POST /backtest/run` | `backtest.py` | Run backtests |
| **AI Mentor** | `/mentor` | `POST /mentor/ask` | `mentor.py` | RAG-based Q&A |
| **Indicators** | `/indicators` | N/A | N/A | Client-side reference |
| **API Status** | Component | `GET /health` | `health.py` | Real-time connectivity check |

### Detailed Backend Endpoints

#### Health & System
- **`GET /`** - API welcome message
- **`GET /health`** - System health check with component status

#### Data Management (`/data`)
- **`POST /data/ingest`** - Ingest market data from Yahoo Finance
  - Parameters: `symbol`, `start_date`, `end_date`, `timeframe`
  - Returns: Number of records inserted
- **`GET /data/latest/{symbol}`** - Get latest market data
  - Query params: `timeframe`, `limit`

#### Signal Generation (`/signals`)
- **`POST /signals/generate`** - Generate trading signals
  - Parameters: `symbol`, `timeframe`, `strategy`, `start_date`, `end_date`
  - Returns: List of signals with entry/exit points
- **`GET /signals/strategies`** - List available strategies

#### Backtesting (`/backtest`)
- **`POST /backtest/run`** - Run strategy backtest
  - Parameters: `symbol`, `timeframe`, `strategy`, `initial_capital`, `start_date`, `end_date`
  - Returns: Performance metrics (returns, Sharpe ratio, drawdown, win rate, trades)

#### AI Agents (`/agents`)
- **`POST /agents/analyze`** - Analyze market with AI agents
  - Parameters: `symbol`, `timeframe`, `agent_type` (or "all")
  - Returns: Agent analysis results
- **`GET /agents/list`** - List available AI agents

#### RAG Mentor (`/mentor`)
- **`POST /mentor/ask`** - Ask trading question
  - Parameters: `question`, `context` (optional)
  - Returns: AI-powered answer with sources

#### ML Models (`/ml`)
- **`POST /ml/predict/direction`** - Predict price direction
- **`POST /ml/forecast/volatility`** - Forecast volatility
- **`POST /ml/classify/regime`** - Classify market regime
- **`GET /ml/models/list`** - List available ML models
- **`GET /ml/health`** - ML models health check

#### Configuration (`/config`)
- **`GET /config`** - Get system configuration

---

## 📄 Page-by-Page Guide

### 1. Home Page (`/`)

**Purpose**: System overview and live snapshot

**Features**:
- System architecture visualization
- Live system stats
- Quick navigation cards

**Backend Integration**:
- `GET /health` - Checks system status every 30 seconds

**Usage**:
1. View overall system health
2. Click navigation cards to access features
3. Monitor backend connectivity (bottom-right indicator)

---

### 2. Market Data Dashboard (`/market`)

**Purpose**: View and ingest market data

**Features**:
- Real-time price data
- Historical data fetching
- Technical indicator charts

**Backend Integration**:
- `POST /data/ingest` - Fetch historical data from Yahoo Finance
- `GET /data/latest/{symbol}` - Get recent market data

**Usage**:
1. Enter symbol (e.g., BTCUSD, AAPL)
2. Select date range
3. Choose timeframe (1d, 4h, 1h)
4. Click "Ingest Data" to fetch from Yahoo Finance
5. View charts and data tables

**Example Request**:
```json
POST /data/ingest
{
  "symbol": "BTCUSD",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "timeframe": "1d"
}
```

---

### 3. Strategy Builder (`/strategy`)

**Purpose**: Design trading strategies visually

**Features**:
- Entry rule configuration (indicators, conditions, values)
- Exit rule configuration (stop loss, take profit)
- Live JSON preview
- Strategy export

**Backend Integration**:
- `GET /signals/strategies` - List available pre-built strategies
- `POST /signals/generate` - Generate signals using strategy

**Usage**:
1. **Define Entry Rule**:
   - Select indicator (RSI, SMA, EMA)
   - Choose condition (<, >)
   - Set value threshold
2. **Define Exit Rules**:
   - Set stop loss percentage
   - Set take profit percentage
3. **View JSON**: Copy strategy definition
4. **Test Strategy**: Navigate to Backtest page

**Strategy JSON Format**:
```json
{
  "entry": [{
    "indicator": "RSI",
    "operator": "<",
    "value": 30
  }],
  "exit": {
    "stopLoss": 5,
    "takeProfit": 10
  }
}
```

---

### 4. Backtesting (`/backtest`)

**Purpose**: Test strategy performance on historical data

**Features**:
- Configuration form (symbol, dates, strategy, capital)
- Performance metrics display
- Equity curve visualization
- Risk metrics

**Backend Integration**:
- `POST /backtest/run` - Execute backtest

**Usage**:
1. **Configure Backtest**:
   - Symbol: BTCUSD, AAPL, etc.
   - Timeframe: 1d, 4h, 1h
   - Strategy: sma_crossover, mean_reversion, multi_agent
   - Initial Capital: $100,000 (default)
   - Date Range: Select start and end dates
2. **Run Backtest**: Click "Run Backtest"
3. **View Results**:
   - Total Return: Percentage gain/loss
   - Sharpe Ratio: Risk-adjusted return
   - Max Drawdown: Largest peak-to-trough decline
   - Win Rate: Percentage of winning trades
   - Total Trades: Number of trades executed

**Example Request**:
```json
POST /backtest/run
{
  "symbol": "BTCUSD",
  "timeframe": "1d",
  "strategy": "sma_crossover",
  "initial_capital": 100000,
  "start_date": "2023-01-01",
  "end_date": "2024-01-01"
}
```

**Example Response**:
```json
{
  "total_return": 0.23,
  "sharpe_ratio": 1.85,
  "max_drawdown": -0.15,
  "win_rate": 0.58,
  "total_trades": 45,
  "equity_curve": [...],
  "trades": [...]
}
```

---

### 5. AI Mentor (`/mentor`)

**Purpose**: Get AI-powered trading insights and answers

**Features**:
- Natural language question input
- Context-aware responses
- Source citations
- Trading knowledge base

**Backend Integration**:
- `POST /mentor/ask` - Ask question to RAG system

**Usage**:
1. **Ask Question**: Type trading-related question
   - Examples:
     - "What is a good RSI threshold for oversold conditions?"
     - "Explain the Sharpe ratio"
     - "How do I calculate position sizing?"
2. **View Answer**: AI-powered response with sources
3. **Follow-up**: Ask additional questions for clarification

**Example Request**:
```json
POST /mentor/ask
{
  "question": "What is a good RSI threshold for oversold conditions?",
  "context": "I'm trading Bitcoin on daily timeframe"
}
```

**Example Response**:
```json
{
  "answer": "For cryptocurrency trading on daily timeframes, RSI below 30 is typically considered oversold. However, during strong downtrends, RSI can remain below 30 for extended periods...",
  "sources": ["Technical Analysis Guide", "RSI Documentation"],
  "confidence": 0.92
}
```

---

### 6. Technical Indicators (`/indicators`)

**Purpose**: Reference guide for technical indicators

**Features**:
- Browse 70+ indicators
- View formulas and calculations
- See usage examples
- Copy code snippets

**Backend Integration**:
- None (client-side reference)

**Usage**:
1. Browse indicator categories
2. Click indicator to view details
3. Copy implementation code
4. Use in strategy builder

**Available Indicators**:
- Trend: SMA, EMA, MACD, Bollinger Bands
- Momentum: RSI, Stochastic, CCI
- Volume: OBV, VWAP, MFI
- Volatility: ATR, Standard Deviation

---

## 🔌 API Endpoints Reference

### Quick Reference Table

| Method | Endpoint | Description | Frontend Usage |
|--------|----------|-------------|----------------|
| GET | `/` | Welcome message | Initial load |
| GET | `/health` | System health | Status indicator |
| POST | `/data/ingest` | Fetch market data | Market page |
| GET | `/data/latest/{symbol}` | Get recent data | Market page |
| POST | `/signals/generate` | Generate signals | Strategy builder |
| GET | `/signals/strategies` | List strategies | Strategy dropdown |
| POST | `/backtest/run` | Run backtest | Backtest page |
| POST | `/agents/analyze` | AI agent analysis | Advanced features |
| GET | `/agents/list` | List agents | Agent selection |
| POST | `/mentor/ask` | Ask question | Mentor page |
| POST | `/ml/predict/direction` | Predict price direction | ML features |
| POST | `/ml/forecast/volatility` | Forecast volatility | ML features |
| POST | `/ml/classify/regime` | Classify market regime | ML features |
| GET | `/ml/models/list` | List ML models | ML status |
| GET | `/config` | Get configuration | Settings |

### Authentication

Currently, the API accepts an optional API key via:
- Header: `X-API-Key: your_api_key`
- Environment: `NEXT_PUBLIC_API_KEY` in `.env.local`

**Note**: Authentication is not enforced in development mode.

### Rate Limits

No rate limits currently enforced. For production:
- Recommended: 100 requests per minute per IP
- Backtest operations: 10 per minute (computationally expensive)

---

## 🔧 Troubleshooting

### Backend Issues

#### Issue: "Backend Offline" indicator
**Cause**: Backend server not running or not accessible

**Solution**:
1. Check backend is running: `http://localhost:8000/health`
2. Restart backend server
3. Verify port 8000 is not in use
4. Check firewall settings

#### Issue: 503 Service Unavailable
**Cause**: Trading API not initialized

**Solution**:
1. Check backend logs for initialization errors
2. Verify `config.yaml` exists in `backend/` directory
3. Ensure MongoDB is running (or TinyDB fallback is enabled)

#### Issue: 500 Internal Server Error
**Cause**: Backend processing error

**Solution**:
1. Check backend terminal for error logs
2. Verify input parameters are correct
3. Check API documentation at `/docs`

### Frontend Issues

#### Issue: Page shows 404
**Cause**: Route does not exist

**Solution**:
1. Check URL spelling
2. Verify page exists in `ui/src/app/`
3. Clear Next.js cache: `rm -rf .next`

#### Issue: Build errors
**Cause**: TypeScript or syntax errors

**Solution**:
1. Check terminal for specific error
2. Verify imports are correct
3. Run `npm run build` to see all errors

#### Issue: Data not loading
**Cause**: API request failed

**Solution**:
1. Open browser DevTools (F12) → Network tab
2. Check for failed requests (red)
3. View request/response details
4. Check backend is running

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Network Error" | Cannot reach backend | Start backend server |
| "Trading API not initialized" | Backend startup failed | Check config.yaml and logs |
| "Invalid parameters" | Request validation failed | Check API docs for required fields |
| "No data found" | Symbol/date range has no data | Ingest data first or change parameters |

---

## 📊 Workflow Examples

### Example 1: Test a New Strategy

1. **Define Strategy** (`/strategy`):
   - Entry: RSI < 30
   - Exit: Stop Loss 5%, Take Profit 10%

2. **Ingest Data** (`/market`):
   - Symbol: BTCUSD
   - Period: 2023-01-01 to 2024-01-01
   - Timeframe: 1d

3. **Run Backtest** (`/backtest`):
   - Use same symbol and dates
   - Select strategy: mean_reversion (similar to RSI strategy)
   - View results

4. **Ask Mentor** (`/mentor`):
   - "How can I improve a strategy with 45% win rate?"
   - Get optimization suggestions

### Example 2: Analyze Market Conditions

1. **Ingest Recent Data** (`/market`):
   - Symbol: AAPL
   - Last 3 months
   - Timeframe: 1d

2. **Use ML Models** (API):
   - Predict direction: POST `/ml/predict/direction`
   - Classify regime: POST `/ml/classify/regime`
   - Forecast volatility: POST `/ml/forecast/volatility`

3. **Get AI Agent Analysis**:
   - POST `/agents/analyze` with agent_type: "all"
   - Review sentiment, volatility, and regime analysis

4. **Ask Mentor**:
   - "What does high volatility regime mean for my strategy?"

---

## 🎓 Best Practices

### Data Management
- ✅ Ingest data before running backtests
- ✅ Use appropriate timeframes (1d for swing trading, 1h for day trading)
- ✅ Fetch sufficient historical data (at least 1 year)

### Strategy Design
- ✅ Start simple (single indicator)
- ✅ Test on different market conditions
- ✅ Use proper risk management (stop loss/take profit)
- ✅ Backtest before live trading

### Performance Analysis
- ✅ Don't optimize for past performance (overfitting)
- ✅ Consider transaction costs
- ✅ Validate on out-of-sample data
- ✅ Monitor multiple metrics (not just returns)

### System Usage
- ✅ Monitor backend connectivity
- ✅ Check API logs for errors
- ✅ Use browser DevTools for debugging
- ✅ Keep both servers running

---

## 📝 Additional Resources

### API Documentation
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Spec: http://localhost:8000/openapi.json

### Project Documentation
- `backend/README.md` - Backend setup and architecture
- `backend/documentation/` - Detailed technical docs
- `QUICKSTART.md` - Quick start guide
- `INTEGRATION_COMPLETE.md` - Integration documentation

### External Resources
- Yahoo Finance API
- Technical Analysis Library (TA-Lib)
- MongoDB Documentation
- Next.js Documentation

---

## 🤝 Support

For issues or questions:
1. Check this guide first
2. Review API documentation at `/docs`
3. Check backend logs for errors
4. Review browser console for frontend errors

---

**Version**: 1.0.0  
**Last Updated**: February 2, 2026  
**Platform**: CF-AI-SDE Trading System
