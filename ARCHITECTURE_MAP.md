# CF-AI-SDE Architecture - Backend-Frontend Mapping

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                       │
│                      http://localhost:3000                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST API
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                           │
│                      http://localhost:8000                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
         │  MongoDB    │ │  ML    │ │  AI       │
         │  (Primary)  │ │ Models │ │  Agents   │
         └─────────────┘ └────────┘ └───────────┘
```

## 📍 Frontend Pages → Backend Endpoints Mapping

### 1. Home Page (`/`)
```
┌─────────────────────────┐
│      Home Page          │
│   (System Overview)     │
└───────────┬─────────────┘
            │
            ▼
    GET /health
    ├─ Check system status
    ├─ Component health
    └─ Connectivity test
```

**Frontend Files:**
- `ui/src/app/page.tsx`
- `ui/src/components/ApiStatusIndicator.tsx`

**Backend Files:**
- `backend/api/routers/health.py`

---

### 2. Market Data Page (`/market`)
```
┌─────────────────────────┐
│    Market Data Page     │
│  (Data Visualization)   │
└───────────┬─────────────┘
            │
            ├─── POST /data/ingest
            │    ├─ Fetch from Yahoo Finance
            │    ├─ Store in database
            │    └─ Return record count
            │
            └─── GET /data/latest/{symbol}
                 ├─ Query parameters: timeframe, limit
                 └─ Return market data array
```

**Frontend Files:**
- `ui/src/app/market/page.tsx`
- `ui/src/services/api.ts` (ingestData, getLatestData)

**Backend Files:**
- `backend/api/routers/data.py`
- `backend/Data-inges-fe/main.py`

---

### 3. Strategy Builder (`/strategy`)
```
┌─────────────────────────┐
│   Strategy Builder      │
│ (Visual Rule Designer)  │
└───────────┬─────────────┘
            │
            ├─── GET /signals/strategies
            │    └─ List available strategies
            │
            └─── POST /signals/generate
                 ├─ Input: symbol, timeframe, strategy
                 └─ Output: signal array
```

**Frontend Files:**
- `ui/src/app/strategy/page.tsx`
- `ui/src/services/api.ts` (generateSignals, listStrategies)

**Backend Files:**
- `backend/api/routers/signals.py`
- `backend/quant_strategy/strategies/`

---

### 4. Backtesting Page (`/backtest`)
```
┌─────────────────────────┐
│    Backtest Page        │
│ (Performance Testing)   │
└───────────┬─────────────┘
            │
            └─── POST /backtest/run
                 ├─ Input: symbol, strategy, dates, capital
                 └─ Output:
                     ├─ total_return
                     ├─ sharpe_ratio
                     ├─ max_drawdown
                     ├─ win_rate
                     ├─ total_trades
                     └─ equity_curve
```

**Frontend Files:**
- `ui/src/app/backtest/page.tsx`
- `ui/src/services/api.ts` (runBacktest)

**Backend Files:**
- `backend/api/routers/backtest.py`
- `backend/Backtesting_risk/backtesting.py`

---

### 5. AI Mentor Page (`/mentor`)
```
┌─────────────────────────┐
│     AI Mentor Page      │
│    (Q&A Interface)      │
└───────────┬─────────────┘
            │
            └─── POST /mentor/ask
                 ├─ Input: question, context
                 └─ Output:
                     ├─ answer
                     ├─ sources
                     └─ confidence
```

**Frontend Files:**
- `ui/src/app/mentor/page.tsx`
- `ui/src/services/api.ts` (askMentor)

**Backend Files:**
- `backend/api/routers/mentor.py`
- `backend/RAG_Mentor/mentor/query_engine.py`

---

### 6. Indicators Page (`/indicators`)
```
┌─────────────────────────┐
│   Indicators Page       │
│  (Reference Library)    │
└─────────────────────────┘
            │
            └─── No Backend (Static Content)
```

**Frontend Files:**
- `ui/src/app/indicators/page.tsx`

**Backend Files:**
- None (client-side reference)

---

## 🔌 Complete API Endpoints Map

### Health & System Routes
```
GET  /                     → Root welcome message
GET  /health               → System health check
GET  /config               → Get system configuration
```

### Data Management Routes (`/data`)
```
POST /data/ingest          → Ingest market data from Yahoo Finance
GET  /data/latest/{symbol} → Get latest market data for symbol
```

### Signal Generation Routes (`/signals`)
```
POST /signals/generate     → Generate trading signals
GET  /signals/strategies   → List available strategies
```

### Backtesting Routes (`/backtest`)
```
POST /backtest/run         → Execute strategy backtest
```

### AI Agent Routes (`/agents`)
```
POST /agents/analyze       → Analyze market with AI agents
GET  /agents/list          → List available AI agents
```

### RAG Mentor Routes (`/mentor`)
```
POST /mentor/ask           → Ask trading question
```

### ML Model Routes (`/ml`)
```
POST /ml/predict/direction → Predict price direction
POST /ml/forecast/volatility → Forecast volatility
POST /ml/classify/regime   → Classify market regime
GET  /ml/models/list       → List available ML models
GET  /ml/health            → ML models health check
```

---

## 🗂️ File Structure Map

### Frontend Structure
```
ui/src/
├── app/
│   ├── page.tsx              → Home page (/)
│   ├── market/page.tsx       → Market data (/market)
│   ├── strategy/page.tsx     → Strategy builder (/strategy)
│   ├── backtest/page.tsx     → Backtesting (/backtest)
│   ├── mentor/page.tsx       → AI mentor (/mentor)
│   └── indicators/page.tsx   → Indicators (/indicators)
├── services/
│   └── api.ts                → API client (all endpoints)
├── components/
│   ├── ApiStatusIndicator.tsx → Backend status
│   ├── ErrorBoundary.tsx     → Error handling
│   └── ToastProvider.tsx     → Notifications
├── hooks/
│   ├── useApiStatus.ts       → Backend connectivity
│   └── useToast.ts           → Toast notifications
└── types/
    └── api.ts                → TypeScript types
```

### Backend Structure
```
backend/
├── api/
│   ├── main.py               → FastAPI app
│   ├── dependencies.py       → Dependency injection
│   └── routers/
│       ├── health.py         → GET /health
│       ├── data.py           → POST /data/ingest
│       ├── signals.py        → POST /signals/generate
│       ├── backtest.py       → POST /backtest/run
│       ├── agents.py         → POST /agents/analyze
│       ├── mentor.py         → POST /mentor/ask
│       ├── ml_models.py      → ML model endpoints
│       └── config.py         → GET /config
├── logical_pipe.py           → Main system orchestrator
├── config.yaml               → System configuration
├── Data-inges-fe/            → Data ingestion module
├── ML_Models/                → Machine learning models
├── AI_Agents/                → AI agent system
├── quant_strategy/           → Trading strategies
├── Backtesting_risk/         → Backtesting engine
└── RAG_Mentor/               → RAG knowledge system
```

---

## 🔄 Data Flow Examples

### Example 1: Running a Backtest
```
User (Frontend)
    │
    │ 1. Fill backtest form
    ▼
[Backtest Page]
    │
    │ 2. api.runBacktest(config)
    ▼
[API Client (api.ts)]
    │
    │ 3. POST /backtest/run
    ▼
[Backend Router (backtest.py)]
    │
    │ 4. get_trading_api()
    ▼
[Trading System API]
    │
    ├─ 5a. Load historical data
    │   └─ [Data Pipeline]
    │
    ├─ 5b. Load strategy
    │   └─ [Strategy Engine]
    │
    └─ 5c. Execute backtest
        └─ [Backtesting Engine]
    │
    │ 6. Return results
    ▼
[Backend Response]
    │
    │ 7. Parse JSON
    ▼
[Frontend State]
    │
    │ 8. Display results
    ▼
[User sees metrics]
```

### Example 2: Asking AI Mentor
```
User (Frontend)
    │
    │ 1. Type question
    ▼
[Mentor Page]
    │
    │ 2. api.askMentor({question})
    ▼
[API Client]
    │
    │ 3. POST /mentor/ask
    ▼
[Backend Router (mentor.py)]
    │
    │ 4. get_trading_api()
    ▼
[Trading System API]
    │
    │ 5. Query RAG system
    ▼
[RAG Mentor]
    │
    ├─ 6a. Embed question
    ├─ 6b. Search ChromaDB
    ├─ 6c. Retrieve context
    └─ 6d. Generate answer (LLM)
    │
    │ 7. Return answer + sources
    ▼
[Backend Response]
    │
    │ 8. Display to user
    ▼
[User sees answer]
```

---

## 🔐 Authentication Flow (Optional)

```
Frontend (.env.local)
    │
    │ NEXT_PUBLIC_API_KEY=your_key
    ▼
[API Client (api.ts)]
    │
    │ Request Interceptor
    │ Add header: X-API-Key
    ▼
[Backend Middleware]
    │
    │ Validate API key
    ├─ Valid → Continue
    └─ Invalid → 401 Unauthorized
```

---

## 📊 State Management

### Frontend State
```
[React State]
├── useState          → Component local state
├── useEffect         → Side effects (API calls)
└── Custom Hooks
    ├── useApiStatus  → Backend connectivity
    └── useToast      → Notifications
```

### Backend State
```
[Dependency Injection]
├── trading_api       → Global TradingSystemAPI instance
└── get_trading_api() → Dependency injector
    └── Returns trading_api or raises 503
```

---

## 🚀 Performance Considerations

### Frontend
- **Lazy Loading**: Pages loaded on-demand
- **Caching**: API responses cached in browser
- **Debouncing**: Search inputs debounced
- **Optimistic Updates**: UI updates before API confirmation

### Backend
- **Connection Pooling**: MongoDB connection pool
- **Model Loading**: ML models loaded once at startup
- **Async Processing**: FastAPI async endpoints
- **Database Indexing**: Indexed queries for performance

---

## 🔍 Debugging Map

### Frontend Debug Points
1. **Browser DevTools** → Network tab → API requests
2. **Console** → Error messages and logs
3. **React DevTools** → Component state
4. **API Status Indicator** → Backend connectivity

### Backend Debug Points
1. **Terminal Logs** → FastAPI startup and request logs
2. **`/docs`** → Interactive API testing
3. **`/health`** → Component health status
4. **Database Logs** → MongoDB connection status

---

## 📚 Quick Reference

| I want to... | Frontend Action | Backend Endpoint |
|--------------|----------------|------------------|
| Check system health | View status indicator | GET /health |
| Fetch market data | Market page → Ingest | POST /data/ingest |
| Create strategy | Strategy builder | N/A (client-side) |
| Test strategy | Backtest page | POST /backtest/run |
| Ask a question | Mentor page | POST /mentor/ask |
| Get ML prediction | Use API directly | POST /ml/predict/* |
| Analyze with AI | Use API directly | POST /agents/analyze |

---

**Visual Guide Version**: 1.0.0  
**Last Updated**: February 2, 2026
