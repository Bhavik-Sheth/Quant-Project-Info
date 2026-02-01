# Quick Start Guide - CF-AI-SDE Trading API

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Configuration

1. Copy example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings (optional):
```env
MONGO_URI=mongodb://localhost:27017/
DB_NAME=trading_system
```

> **Note**: MongoDB is optional. The system automatically falls back to TinyDB if MongoDB is unavailable.

## Running the API

### Option 1: Using uvicorn directly
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### Option 2: Using Python
```bash
cd backend
python -m uvicorn api.main:app --reload --port 8000
```

### Option 3: Using the main script
```bash
cd backend
python api/main.py
```

## Access Points

Once running, access:
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Quick Test

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. Ingest Sample Data
```bash
curl -X POST "http://localhost:8000/data/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "interval": "1d"
  }'
```

### 3. Generate Signals
```bash
curl -X POST "http://localhost:8000/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "strategy": "rsi",
    "lookback_period": 90
  }'
```

### 4. Run Backtest
```bash
curl -X POST "http://localhost:8000/backtest/run" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "strategy": "macd",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000
  }'
```

## Using the Interactive Docs

1. Open http://localhost:8000/docs
2. Click on any endpoint to expand
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"
6. View the response

## Available Endpoints

### Health & Info
- `GET /` - Welcome message
- `GET /health` - System health status

### Data Management
- `POST /data/ingest` - Fetch and store market data
- `GET /data/latest/{symbol}` - Retrieve latest data

### Signal Generation
- `POST /signals/generate` - Generate trading signals
- `GET /signals/strategies` - List available strategies

### Backtesting
- `POST /backtest/run` - Run strategy backtest

### AI Agents
- `POST /agents/analyze` - Run AI agent analysis
- `GET /agents/list` - List available agents

### RAG Mentor
- `POST /mentor/ask` - Ask trading questions

### Configuration
- `GET /config/` - Get system configuration

## Strategies Available

- `rsi` - RSI-based mean reversion
- `macd` - MACD crossover strategy
- `ml_enhanced` - ML model predictions
- `multi_agent` - Multi-agent ensemble

## Agent Types

- `market_data` - Data quality analysis
- `risk` - Risk assessment
- `sentiment` - Sentiment analysis
- `volatility` - Volatility forecasting
- `regime` - Market regime detection
- `full` - Complete multi-agent analysis

## Troubleshooting

### MongoDB Connection Failed
- **Don't worry!** The system automatically falls back to TinyDB
- Data stored in `data/fallback/*.json`
- No functionality is lost

### Import Errors
```bash
cd backend
python test_imports.py
```

### Module Not Found
```bash
pip install -r requirements.txt
```

### Port Already in Use
```bash
uvicorn api.main:app --reload --port 8001
```

## Development Mode

Enable auto-reload for development:
```bash
uvicorn api.main:app --reload --log-level debug
```

## Production Mode

For production deployment:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

Run import tests:
```bash
python test_imports.py
```

Expected output:
```
✅ Passed: 46/46
🎉 All imports successful! Backend is ready for FastAPI.
```

## Common Issues

### 1. Config file not found
- Ensure `config.yaml` exists in backend directory
- Check example config files

### 2. API returns 503
- TradingSystemAPI not initialized
- Check startup logs for errors

### 3. Slow first request
- ML models loading on first use
- Subsequent requests will be faster

## Next Steps

1. **Explore API Docs**: Visit http://localhost:8000/docs
2. **Test Endpoints**: Use interactive documentation
3. **Ingest Data**: Start with a small date range
4. **Generate Signals**: Try different strategies
5. **Run Backtests**: Evaluate strategy performance
6. **Ask Mentor**: Query trading knowledge base

## Support

For issues:
1. Check `documentation/API_RESTRUCTURE_SUMMARY.md`
2. Review logs in terminal
3. Test imports with `python test_imports.py`
4. Verify configuration in `config.yaml`

---

**Happy Trading! 🚀📈**
