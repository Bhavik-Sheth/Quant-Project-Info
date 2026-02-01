# Quick Start Guide - Full Stack Integration

## Prerequisites
- Python 3.8+ installed
- Node.js 18+ installed
- MongoDB running (optional - TinyDB fallback available)

## Step 1: Start Backend

```powershell
# Navigate to backend
cd backend

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --reload --port 8000
```

**Expected Output:**
```
🚀 CF-AI-SDE Trading System API Starting...
✅ Trading System API initialized successfully
✅ database: ok
✅ data_ingestion: ok
✅ ml_models: ok
✅ agents: ok
📚 API Documentation: http://localhost:8000/docs
🔧 Health Check: http://localhost:8000/health
```

**Verify Backend:**
- Open browser: http://localhost:8000/docs
- You should see the FastAPI Swagger UI with all endpoints including the new ML models section

## Step 2: Start Frontend

```powershell
# Navigate to UI folder (from project root)
cd ui

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

**Expected Output:**
```
  ▲ Next.js 16.1.4
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

**Verify Frontend:**
- Open browser: http://localhost:3000
- Check bottom-right corner for API status indicator
- Should show 🟢 "Backend Connected"

## Step 3: Test the Integration

### Test 1: Health Check
1. Click the API status indicator (bottom-right)
2. Should show "Backend Connected" with green dot
3. Shows last check time

### Test 2: Run a Backtest
1. Navigate to Backtest page (from sidebar or menu)
2. Fill in the form:
   - **Symbol:** BTCUSD
   - **Timeframe:** 1d
   - **Strategy:** sma_crossover
   - **Initial Capital:** 100000
   - **Start Date:** 2023-01-01
   - **End Date:** 2024-01-01
3. Click "Run Backtest"
4. Wait for the process (toast notification will appear)
5. Results will display with metrics and equity curve

### Test 3: Check ML Models
Open a browser and test the ML endpoints directly:

**List Available Models:**
```
GET http://localhost:8000/ml/models/list
```

Should return:
```json
{
  "models": [
    {
      "name": "XGBoost Direction Predictor",
      "type": "direction",
      "status": "available"
    },
    ...
  ]
}
```

### Test 4: Error Handling
1. Stop the backend server (Ctrl+C)
2. Refresh frontend
3. API status indicator should show 🔴 "Backend Offline"
4. Try to run a backtest
5. Should see error toast notification: "Network error - cannot connect to backend"

## Step 4: Explore the API

### Using Swagger UI (Recommended for Testing)
1. Open http://localhost:8000/docs
2. Expand any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. See response below

### Available Endpoint Categories

**Health & System**
- GET /health - Check system health

**Data Management**
- POST /data/ingest - Ingest market data
- GET /data/latest - Get latest data

**ML Models (NEW!)**
- POST /ml/predict/direction - Predict market direction
- POST /ml/forecast/volatility - Forecast volatility
- POST /ml/classify/regime - Classify market regime
- POST /ml/train/direction - Train direction model
- POST /ml/train/volatility - Train volatility model
- POST /ml/train/regime - Train regime model
- GET /ml/models/list - List all models
- GET /ml/health - ML service health

**Signals**
- POST /signals/generate - Generate trading signals
- GET /signals/strategies - List strategies

**Backtesting**
- POST /backtest/run - Run backtest

**AI Agents**
- POST /agents/analyze - Analyze with agents
- GET /agents/list - List available agents

**RAG Mentor**
- POST /mentor/ask - Ask trading mentor

**Configuration**
- GET /config - Get system configuration

## Common Issues

### Issue 1: Port Already in Use
**Error:** "Address already in use"

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with the number from above)
taskkill /PID <PID> /F

# Or use a different port
uvicorn api.main:app --reload --port 8001
```

Then update `ui/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Issue 2: Module Not Found
**Error:** "ModuleNotFoundError: No module named 'fastapi'"

**Solution:**
```powershell
cd backend
pip install -r requirements.txt
```

### Issue 3: Frontend Can't Connect
**Error:** API status shows "Backend Offline"

**Checklist:**
1. Is backend running? Check http://localhost:8000/health
2. Check `.env.local` has correct URL
3. Is CORS enabled? (It is by default)
4. Try manual refresh on status indicator

### Issue 4: Toast Notifications Not Showing
**Solution:**
```powershell
# Reinstall dependencies
cd ui
npm install sonner --save
npm run dev
```

## Environment Variables

### Backend (`backend/.env.example`)
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=trading_db
GEMINI_API_KEY=your_key_here
# ... other keys
```

### Frontend (`ui/.env.local`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=trading_db
```

## Development Workflow

### Making Changes to Backend
1. Edit files in `backend/`
2. Save (uvicorn will auto-reload)
3. Check terminal for errors
4. Test in Swagger UI or frontend

### Making Changes to Frontend
1. Edit files in `ui/src/`
2. Save (Next.js will hot-reload)
3. Check browser console for errors
4. Test in browser

## Production Deployment

### Backend
```bash
# Use production server
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd ui
npm run build
npm start
```

### Environment Configuration for Production
Update CORS in `backend/api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Production domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Update `ui/.env.production`:
```env
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

## Next Steps

1. **Add Data**: Use the data ingestion endpoint to add market data
2. **Train Models**: Train ML models on your data
3. **Create Strategies**: Build custom trading strategies
4. **Run Backtests**: Test your strategies on historical data
5. **Analyze with Agents**: Use AI agents for market analysis
6. **Ask Mentor**: Get trading advice from the RAG mentor

## Support & Documentation

- **API Documentation**: http://localhost:8000/docs
- **Integration Guide**: See `INTEGRATION_COMPLETE.md`
- **Backend Docs**: See `backend/documentation/`
- **Issue Tracker**: Create issues in your repository

---

**Status: Ready for Development** 🚀

All systems integrated and operational!
