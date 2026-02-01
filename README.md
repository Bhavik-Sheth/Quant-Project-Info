# CF-AI-SDE: AI-Powered Multi-Agent Trading System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.13+-green.svg)
![Next.js](https://img.shields.io/badge/next.js-16.1.4-black.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

**A sophisticated algorithmic trading platform combining Machine Learning, AI Agents, and Real-time Analytics**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

CF-AI-SDE (Cryptocurrency & Financial Markets - AI Software Development Environment) is a comprehensive trading system that leverages artificial intelligence, machine learning, and multi-agent architectures to provide sophisticated market analysis, strategy backtesting, and automated trading capabilities.

### Key Highlights

- **7 Specialized AI Agents** for market analysis (sentiment, volatility, risk, regime detection)
- **3 Production ML Models** (Direction Prediction, Volatility Forecasting, Regime Classification)
- **70+ Technical Indicators** with real-time feature engineering
- **Real-time Backtesting Engine** with realistic execution simulation
- **RAG-powered Trading Mentor** for knowledge-based decision support
- **Modern Web Interface** built with Next.js and React
- **RESTful API** with interactive documentation
- **Multi-database Support** (MongoDB primary, TinyDB fallback)

---

## ✨ Features

### 🤖 AI & Machine Learning

- **Direction Prediction**: XGBoost-based model for price movement forecasting
- **Volatility Forecasting**: GARCH and LSTM models for volatility prediction
- **Regime Classification**: Market regime detection (bull, bear, sideways, high volatility)
- **Ensemble Methods**: Weighted agent consensus for signal generation
- **Adaptive Learning**: Model retraining pipeline with performance tracking

### 📊 Data Management

- **Multi-source Data Ingestion**: Yahoo Finance, Alpaca, Custom APIs
- **Real-time & Historical Data**: Support for multiple timeframes (1m to 1M)
- **Data Validation**: Automated quality checks and outlier detection
- **Feature Engineering**: 70+ technical indicators calculated on-the-fly
- **Data Normalization**: Robust scaling and feature selection

### 🎯 Strategy Development

- **Visual Strategy Builder**: Design strategies without coding
- **Pre-built Strategies**: SMA Crossover, Mean Reversion, ML Enhanced, Multi-Agent
- **Custom Strategy Support**: Declarative strategy definitions with JSON
- **Risk Management**: Position sizing, stop-loss, take-profit automation
- **Portfolio Management**: Multi-asset portfolio optimization

### 📈 Backtesting & Analysis

- **Realistic Execution**: Slippage, transaction costs, latency simulation
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor
- **Equity Curve Visualization**: Interactive charts with drill-down capabilities
- **Trade Analysis**: Individual trade breakdown with entry/exit analysis
- **Walk-Forward Testing**: Out-of-sample validation

### 🧠 RAG Trading Mentor

- **Knowledge Base**: Curated trading principles and market insights
- **Context-aware Responses**: ChromaDB vector search for relevant information
- **Multi-LLM Support**: Gemini, Groq, OpenAI integration
- **Conversation Memory**: Session-based context retention
- **Performance Coaching**: Strategy improvement suggestions

### 🌐 Web Interface

- **Modern UI/UX**: Responsive design with dark theme
- **Real-time Updates**: Live market data and system status
- **Interactive Charts**: Candlestick, line, and technical indicator overlays
- **Dashboard Analytics**: System health and performance metrics
- **Mobile-ready**: Works on desktop, tablet, and mobile devices

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js + React)                    │
│         TypeScript, Tailwind CSS, Framer Motion, Sonner         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST API
┌────────────────────────────┴────────────────────────────────────┐
│                     Backend (FastAPI + Python)                   │
│                         logical_pipe.py                          │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│ Data        │ ML Models    │ AI Agents    │ Strategy Engine    │
│ Ingestion   │              │              │                    │
│ ├─Yahoo     │ ├─XGBoost    │ ├─Sentiment  │ ├─Technical        │
│ ├─Alpaca    │ ├─LSTM       │ ├─Volatility │ ├─ML Enhanced      │
│ └─CSV       │ └─GARCH      │ ├─Risk       │ ├─Multi-Agent      │
│             │              │ ├─Regime     │ └─Options          │
│             │              │ ├─Market Data│                    │
│             │              │ └─Signal Agg │                    │
└─────────────┴──────────────┴──────────────┴────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      Data Persistence Layer                      │
│  MongoDB (Primary) • TinyDB (Fallback) • ChromaDB (Vectors)    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Next.js 16, React 19, TypeScript | User interface and visualization |
| **Backend API** | FastAPI, Python 3.13 | RESTful API and orchestration |
| **Data Pipeline** | pandas, yfinance, numpy | Data ingestion and processing |
| **ML Models** | XGBoost, TensorFlow, scikit-learn | Prediction and classification |
| **AI Agents** | Pydantic, Custom Framework | Market analysis and decision making |
| **Strategy Engine** | Custom Python | Strategy execution and backtesting |
| **RAG System** | ChromaDB, Gemini/Groq | Knowledge retrieval and Q&A |
| **Databases** | MongoDB, TinyDB, ChromaDB | Data persistence and vector storage |

---

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.13+, pip, MongoDB (optional)
- **Frontend**: Node.js 16+, npm or yarn
- **Optional**: MongoDB for production, Docker for containerization

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/janvi-0706/CF-AI-SDE.git
cd CF-AI-SDE
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# Setup database (optional - creates MongoDB indexes)
python setup_database.py

# Start backend server
uvicorn api.main:app --reload --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

#### 3. Frontend Setup

```bash
cd ui

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with backend URL

# Start development server
npm run dev
```

Frontend will be available at:
- Application: http://localhost:3000

### Verify Installation

1. Check backend health: http://localhost:8000/health
2. Check API docs: http://localhost:8000/docs
3. Open frontend: http://localhost:3000
4. Verify API status indicator (bottom-right) shows green "Backend Connected"

---

## 💻 Usage

### 1. Ingest Market Data

```bash
# Via UI: Navigate to /market page and use the ingest form

# Via API:
curl -X POST "http://localhost:8000/data/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "timeframe": "1d"
  }'
```

### 2. Generate Trading Signals

```bash
# Via API:
curl -X POST "http://localhost:8000/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "timeframe": "1d",
    "strategy": "ml_enhanced",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }'
```

### 3. Run Backtest

```bash
# Via UI: Navigate to /backtest page and fill the form

# Via API:
curl -X POST "http://localhost:8000/backtest/run" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSD",
    "timeframe": "1d",
    "strategy": "sma_crossover",
    "initial_capital": 100000,
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }'
```

### 4. Ask Trading Mentor

```bash
# Via UI: Navigate to /mentor page

# Via API:
curl -X POST "http://localhost:8000/mentor/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a good RSI threshold for oversold conditions?",
    "context": "Trading Bitcoin on daily timeframe"
  }'
```

### 5. Use AI Agents

```python
from logical_pipe import TradingSystemAPI

# Initialize system
api = TradingSystemAPI("config.yaml")

# Analyze with all agents
results = api.agent_orchestrator.analyze(
    symbol="AAPL",
    timeframe="1d",
    agent_type="all"
)

# Get consensus signal
consensus = api.agent_orchestrator.get_consensus_signal(results)
print(f"Signal: {consensus['action']}, Confidence: {consensus['confidence']}")
```

---

## 📁 Project Structure

```
CF-AI-SDE/
├── backend/                    # Python backend
│   ├── api/                   # FastAPI application
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── dependencies.py   # Dependency injection
│   │   ├── routers/          # API route handlers
│   │   └── models/           # Pydantic models
│   ├── AI_Agents/            # Multi-agent system
│   │   ├── agents.py         # Agent implementations
│   │   └── communication_protocol.py
│   ├── ML_Models/            # Machine learning models
│   │   ├── direction_pred.py
│   │   ├── Volatility_Forecasting.py
│   │   └── Regime_Classificaiton.py
│   ├── Data-inges-fe/        # Data ingestion & feature engineering
│   │   ├── main.py
│   │   └── src/
│   ├── quant_strategy/       # Trading strategies
│   │   ├── strategies/
│   │   ├── engine.py
│   │   └── base.py
│   ├── Backtesting_risk/     # Backtesting engine
│   │   ├── backtesting.py
│   │   └── analysis.py
│   ├── RAG_Mentor/           # RAG system
│   │   ├── mentor/
│   │   ├── vector_db/
│   │   └── knowledge/
│   ├── db/                   # Database utilities
│   │   ├── connection.py
│   │   ├── readers.py
│   │   └── writers.py
│   ├── data/                 # Data storage
│   │   ├── raw/             # Raw market data
│   │   ├── validated/       # Cleaned data
│   │   └── features/        # Engineered features
│   ├── logical_pipe.py      # Main orchestrator
│   ├── config.yaml          # System configuration
│   └── requirements.txt     # Python dependencies
│
├── ui/                       # Next.js frontend
│   ├── src/
│   │   ├── app/             # Pages (App Router)
│   │   │   ├── page.tsx    # Home page
│   │   │   ├── market/     # Market data page
│   │   │   ├── strategy/   # Strategy builder
│   │   │   ├── backtest/   # Backtesting page
│   │   │   └── mentor/     # AI mentor page
│   │   ├── components/      # React components
│   │   │   ├── ApiStatusIndicator.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   └── ToastProvider.tsx
│   │   ├── services/        # API client
│   │   │   └── api.ts
│   │   ├── hooks/           # Custom React hooks
│   │   │   ├── useApiStatus.ts
│   │   │   └── useToast.ts
│   │   └── types/           # TypeScript types
│   │       └── api.ts
│   ├── package.json         # Node dependencies
│   ├── next.config.ts       # Next.js config
│   └── .env.local           # Environment variables
│
├── consistency-checker/      # Code quality tools
├── integration/              # Integration utilities
├── repo-documenter/          # Documentation generator
│
├── .gitignore
├── README.md                 # This file
├── USER_GUIDE.md            # Comprehensive user guide
├── ARCHITECTURE_MAP.md      # Visual architecture guide
├── INTEGRATION_COMPLETE.md  # Integration documentation
└── QUICKSTART.md            # Quick start guide
```

---

## ⚙️ Configuration

### Backend Configuration (`backend/config.yaml`)

```yaml
system:
  name: "CF-AI-SDE"
  version: "1.0.0"
  environment: "development"
  log_level: "INFO"

data_ingestion:
  source: "yahoo"
  symbols: ["AAPL", "MSFT", "GOOGL"]
  default_timeframe: "1d"
  lookback_days: 365

ml_models:
  direction:
    default_model: "xgboost"
  volatility:
    default_model: "garch"
  regime:
    default_model: "hmm"

ai_agents:
  enabled: ["sentiment", "volatility", "risk", "regime"]
  consensus_threshold: 0.6

api_keys:
  gemini: "${GEMINI_API_KEY}"
  groq: "${GROQ_API_KEY}"
  alpaca_key: "${ALPACA_API_KEY}"
  alpaca_secret: "${ALPACA_API_SECRET}"
```

### Frontend Configuration (`ui/.env.local`)

```env
# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=

# MongoDB (Optional - for direct DB access)
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=trading_db
```

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

# Database
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=cf_ai_sde

# System
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## 📚 API Documentation

### Interactive API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Quick API Reference

#### Health & System

```bash
GET  /                    # Welcome message
GET  /health              # System health check
GET  /config              # Get configuration
```

#### Data Management

```bash
POST /data/ingest         # Ingest market data
GET  /data/latest/{symbol} # Get latest data
```

#### Signal Generation

```bash
POST /signals/generate    # Generate trading signals
GET  /signals/strategies  # List available strategies
```

#### Backtesting

```bash
POST /backtest/run        # Run strategy backtest
```

#### AI Agents

```bash
POST /agents/analyze      # Analyze with AI agents
GET  /agents/list         # List available agents
```

#### RAG Mentor

```bash
POST /mentor/ask          # Ask trading question
```

#### ML Models

```bash
POST /ml/predict/direction     # Predict price direction
POST /ml/forecast/volatility   # Forecast volatility
POST /ml/classify/regime       # Classify market regime
GET  /ml/models/list           # List ML models
```

For detailed API documentation with request/response examples, see [USER_GUIDE.md](USER_GUIDE.md).

---

## 🔧 Development

### Running in Development Mode

#### Backend

```bash
cd backend

# With auto-reload
uvicorn api.main:app --reload --port 8000

# With specific host
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# With log level
uvicorn api.main:app --reload --log-level debug
```

#### Frontend

```bash
cd ui

# Development server with Turbopack
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

### Code Quality Tools

#### Consistency Checker

```bash
python consistency-checker/scripts/check_consistency.py
```

#### Python Linting

```bash
cd backend
flake8 .
black .
mypy .
```

#### TypeScript Linting

```bash
cd ui
npm run lint
tsc --noEmit
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_ml_models.py

# Verbose output
pytest -v
```

### Frontend Tests

```bash
cd ui

# Run tests
npm test

# With coverage
npm test -- --coverage

# Watch mode
npm test -- --watch
```

### Integration Tests

```bash
# Test full pipeline
python backend/test_imports.py

# Test backtest engine
python backend/Backtesting_risk/test_backtest.py
```

---

## 🚢 Deployment

### Docker Deployment

#### Build Images

```bash
# Backend
docker build -t cf-ai-sde-backend ./backend

# Frontend
docker build -t cf-ai-sde-frontend ./ui
```

#### Run with Docker Compose

```bash
docker-compose up -d
```

### Production Deployment

#### Backend (with Gunicorn)

```bash
cd backend
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### Frontend (with PM2)

```bash
cd ui
npm run build
pm2 start npm --name "cf-ai-sde-ui" -- start
```

### Environment-specific Configuration

```yaml
# production config.yaml
system:
  environment: "production"
  log_level: "WARNING"

# Use environment variables for secrets
api_keys:
  gemini: "${GEMINI_API_KEY}"
  groq: "${GROQ_API_KEY}"
```

---

## 📖 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Comprehensive user guide with usage examples
- **[ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md)** - Visual architecture and data flow diagrams
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** - Integration documentation
- **[backend/documentation/](backend/documentation/)** - Detailed technical documentation
- **API Docs**: http://localhost:8000/docs (when running)

### Module-specific Documentation

- **[AI_Agents/README.md](backend/AI_Agents/README.md)** - AI agent system
- **[ML_Models/Models_Documentation.md](backend/ML_Models/Models_Documentation.md)** - ML models
- **[RAG_Mentor/README.md](backend/RAG_Mentor/README.md)** - RAG system
- **[Backtesting_risk/README.md](backend/Backtesting_risk/README.md)** - Backtesting engine
- **[Data-inges-fe/README.md](backend/Data-inges-fe/README.md)** - Data pipeline

---

## 🔍 Troubleshooting

### Common Issues

#### Backend won't start

```bash
# Check Python version
python --version  # Should be 3.13+

# Check dependencies
pip list

# Check config file
ls backend/config.yaml

# Check logs
tail -f backend/logs/*.log
```

#### Frontend shows "Backend Offline"

1. Verify backend is running: http://localhost:8000/health
2. Check `.env.local` has correct `NEXT_PUBLIC_API_URL`
3. Check browser console for CORS errors
4. Verify firewall isn't blocking port 8000

#### 503 Service Unavailable

- Backend is running but not initialized properly
- Check backend logs for initialization errors
- Verify `config.yaml` exists and is valid
- Ensure MongoDB is running (or TinyDB fallback is enabled)

#### ImportError in Python modules

```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/CF-AI-SDE/backend"

# On Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\CF-AI-SDE\backend
```

For more troubleshooting tips, see [USER_GUIDE.md#troubleshooting](USER_GUIDE.md#troubleshooting).

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
```bash
git clone https://github.com/your-username/CF-AI-SDE.git
cd CF-AI-SDE
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Follow existing code style
- Add tests for new features
- Update documentation

3. **Test Your Changes**
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd ui && npm test

# Lint
python -m flake8
npm run lint
```

4. **Submit Pull Request**
- Write clear commit messages
- Reference related issues
- Include test results

### Development Guidelines

- **Code Style**: Follow PEP 8 (Python), Airbnb (TypeScript)
- **Commits**: Use conventional commits (feat, fix, docs, etc.)
- **Documentation**: Update docs for new features
- **Tests**: Maintain >80% code coverage
- **Security**: Never commit API keys or secrets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data**: Yahoo Finance API
- **ML Libraries**: XGBoost, TensorFlow, scikit-learn
- **Web Framework**: FastAPI, Next.js
- **UI Components**: Tailwind CSS, Framer Motion
- **Vector Database**: ChromaDB
- **LLM Integration**: Google Gemini, Groq

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/janvi-0706/CF-AI-SDE/issues)
- **Documentation**: [USER_GUIDE.md](USER_GUIDE.md)
- **Email**: support@cf-ai-sde.com

---

## 🗺️ Roadmap

### Version 1.1 (Q2 2026)
- [ ] Real-time WebSocket support
- [ ] Additional ML models (Transformer, GNN)
- [ ] Advanced portfolio optimization
- [ ] Mobile app (React Native)

### Version 2.0 (Q4 2026)
- [ ] Live trading integration (Alpaca, Interactive Brokers)
- [ ] Multi-strategy ensemble
- [ ] Advanced risk analytics
- [ ] Cloud deployment templates

---

## 📊 Project Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-85%25-yellow.svg)
![Coverage](https://img.shields.io/badge/coverage-78%25-orange.svg)
![Docs](https://img.shields.io/badge/docs-100%25-brightgreen.svg)

**Last Updated**: February 2, 2026  
**Version**: 1.0.0  
**Status**: Active Development

---

<div align="center">

Made with ❤️ by the CF-AI-SDE Team

[⬆ Back to Top](#cf-ai-sde-ai-powered-multi-agent-trading-system)

</div>
