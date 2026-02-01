# CF-AI-SDE Backend - Quick Start Guide

## 🎉 Status: All Imports Working!

**Test Result**: 26/26 imports passed ✅  
**Status**: Ready for FastAPI development

---

## Prerequisites

1. **Python 3.10+** installed
2. **Virtual environment** activated (conda/venv)
3. **MongoDB** installed (optional - has fallback)
4. **API Keys** in `.env` file

---

## Quick Start (3 Steps)

### 1. Verify Imports
```bash
cd backend
python test_imports.py
```

Expected: "🎉 All imports successful!"

### 2. Check System Health
```bash
python -c "from logical_pipe import TradingSystemAPI, ConfigLoader; api = TradingSystemAPI(ConfigLoader('config.yaml')); print(api.health_check())"
```

Expected: Dictionary with component statuses

### 3. Start FastAPI Server
```bash
pip install fastapi uvicorn  # if not installed
uvicorn api:app --reload --port 8000
```

Then visit: **http://localhost:8000/docs**

---

## API Endpoints Available

### 🏥 Health & Status
- `GET /` - API welcome message
- `GET /health` - System health check

### 📊 Data Ingestion
- `POST /data/ingest` - Fetch & store market data
- `GET /data/latest/{symbol}` - Get latest data from DB

### 🎯 Signal Generation
- `POST /signals/generate` - Generate trading signals
  - Strategies: `rsi`, `macd`, `ml_enhanced`, `multi_agent`

### 🧪 Backtesting
- `POST /backtest/run` - Run strategy backtest with metrics

### 🤖 AI Agents
- `POST /agents/run` - Run agent analysis
  - Types: `market_data`, `risk`, `sentiment`, `volatility`, `regime`, `full`

### 🎓 RAG Mentor
- `POST /mentor/ask` - Ask trading mentor for guidance

### 🔧 Development
- `GET /config` - View system configuration (debug)

---

## Example API Calls

### Health Check
```bash
curl http://localhost:8000/health
```

### Ingest Market Data
```bash
curl -X POST http://localhost:8000/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "interval": "1d"
  }'
```

### Generate Signals
```bash
curl -X POST http://localhost:8000/signals/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "strategy": "rsi",
    "lookback_period": 100
  }'
```

### Run Backtest
```bash
curl -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "strategy": "rsi",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000.0
  }'
```

### Ask RAG Mentor
```bash
curl -X POST "http://localhost:8000/mentor/ask?question=How%20to%20manage%20risk%3F&symbol=AAPL"
```

---

## Project Structure

```
backend/
├── api.py                    # FastAPI entry point (NEW)
├── logical_pipe.py           # Unified backend API
├── config.yaml               # System configuration
├── .env                      # Environment variables
├── test_imports.py           # Import verification (NEW)
├── IMPORT_FIX_SUMMARY.md     # Fix documentation (NEW)
├── OPTIONAL_PACKAGES.md      # Missing packages guide (NEW)
├── QUICKSTART.md             # This file (NEW)
│
├── AI_Agents/                # Multi-agent system
├── ML_Models/                # ML models (direction, volatility, regime, GAN)
├── quant_strategy/           # Strategy framework & backtesting
├── Backtesting_risk/         # Risk management & execution
├── Data-inges-fe/            # Data ingestion & feature engineering
├── RAG_Mentor/               # RAG-based trading mentor
├── db/                       # MongoDB integration
└── setup_database.py         # DB initialization script
```

---

## Configuration Files

### .env (Backend Root)
Contains API keys for:
- OpenAI, Gemini, Groq, Hugging Face
- News API, Alpha Vantage, FRED
- MongoDB connection string

### config.yaml (Backend Root)
Contains:
- Model configurations
- Agent settings
- Strategy parameters
- Risk management rules
- Data sources

---

## MongoDB Setup (Optional)

System works without MongoDB using fallbacks, but for full functionality:

### Option A: Local MongoDB
```bash
# Install MongoDB Community Edition
# Windows: https://www.mongodb.com/try/download/community
# Mac: brew install mongodb-community
# Linux: sudo apt install mongodb

# Start MongoDB
mongod --dbpath /data/db

# Initialize database
cd backend
python setup_database.py
```

### Option B: MongoDB Atlas (Cloud)
1. Create account at mongodb.com/cloud/atlas
2. Create free M0 cluster
3. Get connection string
4. Update `.env`: `MONGODB_URI=mongodb+srv://...`

**Note**: Atlas free tier has 512MB limit - may be insufficient for extensive historical data.

---

## Optional Packages

Install these for additional features:

```bash
# GARCH volatility models
pip install arch

# Carbon emissions tracking
pip install codecarbon

# Advanced orchestration
pip install langchain langgraph langchain-community
```

See [OPTIONAL_PACKAGES.md](OPTIONAL_PACKAGES.md) for details.

---

## Troubleshooting

### Import Errors
```bash
# Verify all imports work
python test_imports.py

# If errors, check:
1. Are you in the backend/ directory?
2. Is virtual environment activated?
3. Are required packages installed?
```

### MongoDB Connection Issues
```bash
# Check MongoDB is running
mongosh  # or mongo

# Verify connection string in .env
echo $MONGODB_URI

# Test connection
python -c "from db.connection import get_connection; print(get_connection())"
```

### API Key Issues
```bash
# Verify .env file exists
ls -la .env

# Check required keys are set
cat .env | grep API_KEY

# Test specific key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

### FastAPI Not Starting
```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Check for port conflicts
lsof -i :8000  # Unix/Mac
netstat -ano | findstr :8000  # Windows

# Run with different port
uvicorn api:app --port 8001
```

---

## Development Workflow

### 1. Test New Features
```bash
# Write test in test_imports.py or create new test file
python test_new_feature.py
```

### 2. Update logical_pipe.py
```python
# Add new method to TradingSystemAPI class
def new_feature(self, params):
    # Implementation
    pass
```

### 3. Add FastAPI Route
```python
# In api.py
@app.post("/new-feature", tags=["Category"])
async def new_feature_endpoint(request: RequestModel):
    result = trading_api.new_feature(request.params)
    return {"status": "success", "data": result}
```

### 4. Test API
```bash
# Auto-reload is enabled with --reload flag
# Just save files and FastAPI will restart automatically

# Test endpoint
curl -X POST http://localhost:8000/new-feature \
  -H "Content-Type: application/json" \
  -d '{"params": "value"}'
```

---

## Next Steps

1. ✅ **Verify all imports** - `python test_imports.py`
2. ✅ **Start FastAPI** - `uvicorn api:app --reload`
3. 📖 **Read API docs** - http://localhost:8000/docs
4. 🧪 **Test endpoints** - Use Swagger UI or curl
5. 🗄️ **Setup MongoDB** - When ready for persistent storage
6. 🚀 **Build features** - Add routes, expand functionality
7. 🧪 **Write tests** - Create comprehensive test suite
8. 🎨 **Build frontend** - Connect React/Vue to API

---

## Support & Documentation

- **Import Fixes**: [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md)
- **Optional Packages**: [OPTIONAL_PACKAGES.md](OPTIONAL_PACKAGES.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Module Docs**:
  - AI Agents: `AI_Agents/README.md`
  - Quant Strategy: `quant_strategy/Strategy_guide.md`
  - RAG Mentor: `RAG_Mentor/README.md`
  - Data Pipeline: `Data-inges-fe/README.md`
  - ML Models: `ML_Models/Models_Documentation.md`

---

## Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| Imports | ✅ 26/26 | All working with fallbacks |
| FastAPI Template | ✅ Ready | api.py with 12 endpoints |
| MongoDB | ⚠️ Optional | Works without it (fallback) |
| API Keys | ✅ Config | Stored in .env |
| LLM Models | ✅ Updated | Lighter models configured |
| Documentation | ✅ Complete | 4 new docs created |

---

## Quick Commands Reference

```bash
# Verify imports
python test_imports.py

# Start API server
uvicorn api:app --reload

# Test health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs  # Mac
start http://localhost:8000/docs  # Windows
xdg-open http://localhost:8000/docs  # Linux

# Run example script
python logical_pipe.py  # Demo usage

# Setup MongoDB
python setup_database.py
```

---

**🚀 You're all set! Start building your trading system.**
