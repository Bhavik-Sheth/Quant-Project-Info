# CF-AI-SDE: Comprehensive Financial AI Strategy Development Engine

> **An end-to-end quantitative trading system** combining data engineering, machine learning, multi-agent AI, backtesting, and RAG-powered analysis with a unified backend API.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [System Components](#system-components)
- [Integration Layer](#integration-layer)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**CF-AI-SDE** (Computational Finance - AI Strategy Development Engine) is a comprehensive quantitative trading platform that integrates:

1. **Data Pipeline** - Multi-source financial data ingestion and feature engineering
2. **ML Models** - Predictive models for direction, volatility, and regime classification
3. **AI Agents** - Multi-agent system for market analysis and decision-making
4. **Strategy Library** - Modular framework for strategy development and backtesting
5. **RAG Trading Mentor** - AI-powered backtest analysis and improvement suggestions
6. **Unified Backend API** - Single entry point for all system operations via `TradingSystemAPI`

### What Makes This Unique?

- ✅ **Unified Integration** - Single API (`logical_pipe.py`) orchestrating all modules
- ✅ **Temporal Integrity** - Strict lookahead bias prevention across all modules
- ✅ **Multi-Agent Intelligence** - 7 specialized agents collaborating via message passing
- ✅ **Explainable AI** - RAG-based analysis with cited trading principles
- ✅ **Regime-Aware** - Adaptive strategies based on market conditions
- ✅ **Production-Ready** - MongoDB integration, error handling, comprehensive logging
- ✅ **Hybrid Persistence** - MongoDB primary with automatic filesystem fallback

---

## 🚀 Quick Start

### 30-Second Setup

```bash
# 1. Clone and install
cd CF-AI-SDE
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (see things_to_get.md for details)

# 3. Initialize database
python setup_database.py

# 4. Run your first backtest
python -c "
from logical_pipe import TradingSystemAPI

# Initialize unified API
api = TradingSystemAPI('./config.yaml')

# Run full pipeline (data → ML → agents → backtest → analysis)
result = api.run_full_pipeline(
    ticker='AAPL',
    start_date='2024-01-01',
    end_date='2024-01-31',
    strategy_name='RSI_Strategy'
)

print('Sharpe Ratio:', result['backtest_results']['metrics']['sharpe_ratio'])
"
```

**That's it!** You've just run a complete end-to-end trading system.

### For detailed setup, see [things_to_get.md](things_to_get.md)

---

## 🏗️ Architecture

![CF-AI-SDE Architecture](RAG_Mentor/architecture_diagram.png)

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                       TradingSystemAPI (logical_pipe.py)             │
│                        Single Unified Entry Point                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  • run_full_pipeline()     • health_check()                  │  │
│  │  • run_partial_pipeline()  • Temporal Coordinator             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  DataPipeline    │  │  ModelManager    │  │ AgentOrchestrator│
│   (Wrapper)      │  │   (Wrapper)      │  │    (Wrapper)     │
│                  │  │                  │  │                  │
│ • ingest_data()  │  │ • train_model()  │  │ • run_agents()   │
│ • features()     │  │ • predict()      │  │ • aggregate()    │
│ • validate()     │  │ • save/load      │  │ • track_perf()   │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Data-inges-fe/   │  │   ML_Models/     │  │   AI_Agents/     │
│ (Original Module)│  │ (Original Module)│  │ (Original Module)│
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   StrategyEngine     │
                    │      (Wrapper)       │
                    │                      │
                    │ • backtest()         │
                    │ • risk_management()  │
                    │ • signal_generation()│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  AnalysisInterface   │
                    │      (Wrapper)       │
                    │                      │
                    │ • analyze_backtest() │
                    │ • ask_question()     │
                    │ • detect_violations()│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    MongoDB + Files   │
                    │  (Hybrid Persistence)│
                    │                      │
                    │ • market_data_*      │
                    │ • ml_models          │
                    │ • agent_memory       │
                    │ • backtest_results   │
                    └──────────────────────┘
```

### Integration Layer Benefits

The `logical_pipe.py` module provides:

1. **Single Entry Point** - No need to interact with multiple modules directly
2. **Consistent Interface** - Unified API patterns across all operations
3. **Automatic Orchestration** - Handles dependencies between modules
4. **Temporal Safety** - `TemporalCoordinator` prevents look-ahead bias
5. **Graceful Degradation** - Falls back to filesystem if MongoDB unavailable
6. **Health Monitoring** - System-wide health checks via `health_check()`
7. **Centralized Config** - Single `config.yaml` for all modules

---

## 🧩 Integration Layer

The **`logical_pipe.py`** module serves as the unified backend API, orchestrating all system components through wrapper classes and the `TradingSystemAPI` facade.

### Core Classes

#### 1. **TradingSystemAPI** (Main Facade)

**Purpose**: Single entry point for all trading operations.

**Key Methods**:

```python
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI('./config.yaml')

# Full end-to-end pipeline
result = api.run_full_pipeline(
    ticker='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-31',
    strategy_name='RSI_Strategy',
    analyze_results=True  # Optional RAG analysis
)

# Partial pipeline (specific stages only)
result = api.run_partial_pipeline(
    stages=['data_ingestion', 'feature_engineering', 'ml_prediction'],
    ticker='MSFT',
    start_date='2024-01-01',
    end_date='2024-06-30'
)

# System health check
status = api.health_check()
# Returns: {'status': 'healthy', 'mongodb_connected': True, 'modules_loaded': {...}}
```

#### 2. **TemporalCoordinator**

**Purpose**: Enforce temporal integrity and prevent look-ahead bias.

**Key Features**:
- Tracks current timestamp in backtesting
- Validates data access against current time
- Provides audit trail of all data accesses
- Blocks future data from being accessed

```python
coordinator = api.temporal_coordinator

# Advance time (in backtesting)
coordinator.advance_time(new_timestamp='2024-01-15')

# Execute pipeline step with temporal validation
result = coordinator.execute_pipeline_step(
    step_name='ml_prediction',
    func=lambda: model.predict(features),
    timestamp='2024-01-15'
)

# Get audit trail
audit_log = coordinator.get_audit_trail()
```

#### 3. **ConfigLoader**

**Purpose**: Centralized YAML configuration with environment variable support.

**Features**:
- Loads `config.yaml` with nested structure
- Resolves `${ENV_VAR}` placeholders from `.env`
- Validates required sections
- Provides dotted-path access: `config.get('ml_models.direction.model_type')`

```python
loader = ConfigLoader('./config.yaml')
config = loader.get_config()

# Access nested values
db_uri = loader.get('database.mongodb_uri')  # Returns resolved env var
agents_config = loader.get('ai_agents')      # Returns full dict
```

#### 4. **DataPipeline** (Wrapper for Data-inges-fe)

**Purpose**: Unified interface for data ingestion, validation, and feature engineering.

```python
pipeline = api.data_pipeline

# Ingest market data
raw_data = pipeline.ingest_data(
    ticker='GOOGL',
    start_date='2024-01-01',
    end_date='2024-12-31',
    data_sources=['yahoo', 'news', 'macro']
)

# Engineer features (70+ indicators)
features = pipeline.engineer_features(
    ticker='GOOGL',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Validate data quality
validation_report = pipeline.validate_data(
    ticker='GOOGL',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

#### 5. **ModelManager** (Wrapper for ML_Models)

**Purpose**: ML model training, prediction, and persistence with MongoDB integration.

**Features**:
- Train and predict with 6 model types (Logistic, Tree, RF, XGBoost, NN, LSTM)
- Automatic model versioning
- MongoDB + file-based hybrid persistence
- Model metadata tracking (training date, accuracy, features used)

```python
manager = api.model_manager

# Train direction prediction model
model_id = manager.train_direction_model(
    ticker='TSLA',
    start_date='2022-01-01',
    end_date='2023-12-31',
    model_type='xgboost',  # or 'lstm', 'random_forest', etc.
    save_to_db=True
)

# Predict (loads latest model automatically)
prediction = manager.predict_direction(
    ticker='TSLA',
    current_features=latest_features,
    model_type='xgboost'
)
# Returns: {'direction': 1, 'confidence': 0.78, 'model_version': 'v3'}

# Train volatility model
vol_model_id = manager.train_volatility_model(
    ticker='TSLA',
    model_type='garch',  # or 'egarch', 'lstm'
    save_to_db=True
)

# Predict volatility
volatility_forecast = manager.predict_volatility(
    ticker='TSLA',
    model_type='garch',
    horizon=5  # 5-day forecast
)

# Train regime classification model
regime_model_id = manager.train_regime_model(
    ticker='SPY',
    model_type='random_forest',  # or 'lstm'
    save_to_db=True
)

# Classify market regime
current_regime = manager.classify_regime(
    ticker='SPY',
    current_features=latest_features,
    model_type='random_forest'
)
# Returns: {'regime': 'TRENDING_UP', 'confidence': 0.85}
```

#### 6. **AgentOrchestrator** (Wrapper for AI_Agents)

**Purpose**: Coordinate 7 specialized AI agents with performance tracking.

**Features**:
- Initialize agents with API keys
- Run agents in parallel or sequentially
- Aggregate signals with confidence weighting
- Track agent performance over time
- Update agent weights based on accuracy

```python
orchestrator = api.agent_orchestrator

# Initialize all agents
orchestrator.initialize_agents(
    gemini_api_key=os.getenv('GEMINI_API_KEY'),
    groq_api_key=os.getenv('GROQ_API_KEY'),
    news_api_key=os.getenv('NEWS_API_KEY')  # Optional
)

# Run all agents
agent_results = orchestrator.run_all_agents(
    ticker='NVDA',
    current_date='2024-06-15',
    context={'features': features, 'market_data': data}
)
# Returns: {
#     'MarketDataAgent': AgentResponse(...),
#     'RiskMonitoringAgent': AgentResponse(...),
#     'SentimentAgent': AgentResponse(...),
#     ... 5 more agents ...
# }

# Get aggregated signal (weighted by performance)
aggregated_signal = orchestrator.get_aggregated_signal(
    agent_results=agent_results
)
# Returns: {'action': 'BUY', 'confidence': 0.82, 'reasoning': '...'}

# Update agent performance (after trade outcome known)
orchestrator.update_agent_performance(
    agent_name='SentimentAgent',
    prediction_correct=True,
    actual_return=0.03
)
```

#### 7. **StrategyEngine** (Wrapper for quant_strategy)

**Purpose**: Unified backtesting interface for all strategy types.

**Features**:
- Run backtests with risk management
- Support technical, ML-enhanced, and options strategies
- Advanced backtesting with agent integration
- Strategy registration and discovery

```python
engine = api.strategy_engine

# Simple backtest
results = engine.backtest(
    strategy_name='RSI_Strategy',
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    commission=0.001
)
# Returns: {
#     'metrics': {'total_return': 0.18, 'sharpe_ratio': 1.4, ...},
#     'trades': [...],
#     'equity_curve': [...]
# }

# Advanced backtest (with ML models + agents)
advanced_results = engine.advanced_backtest(
    strategy_name='ML_Enhanced_Strategy',
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    use_ml_predictions=True,
    use_agent_signals=True,
    risk_manager_config={
        'max_position_size': 0.20,
        'var_limit': 0.05,
        'drawdown_limit': 0.10
    }
)

# Register custom strategy
from quant_strategy.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, context):
        # Your logic here
        pass

engine.register_strategy('MyCustomStrategy', MyCustomStrategy)
```

#### 8. **AnalysisInterface** (Wrapper for RAG_Mentor)

**Purpose**: AI-powered backtest analysis and conversational Q&A.

**Features**:
- Automatic backtest analysis with RAG
- Principle violation detection
- Improvement suggestions with citations
- Natural language Q&A about performance

```python
interface = api.analysis_interface

# Analyze backtest results
analysis_report = interface.analyze_backtest(
    backtest_results=results,
    ticker='AAPL',
    date_range=('2023-01-01', '2023-12-31')
)
# Returns: {
#     'summary': 'Your strategy achieved...',
#     'strengths': ['Good risk management', ...],
#     'weaknesses': ['High turnover', ...],
#     'violations': [
#         {'principle': 'Stop Loss', 'trades': [...]},
#         {'principle': 'Overtrading', 'trades': [...]}
#     ],
#     'improvements': [
#         {'suggestion': 'Add regime filter', 'reasoning': '...', 'citations': [...]}
#     ]
# }

# Ask questions about performance
answer = interface.ask_question(
    question="Why did my strategy underperform in March 2023?",
    context={'backtest_results': results, 'ticker': 'AAPL'}
)
# Returns: "In March 2023, the market experienced high volatility due to..."

# Detect specific violations
violations = interface.detect_violations(
    trades=results['trades'],
    principles=['averaging_down', 'stop_loss', 'overtrading']
)
```

### Configuration System

**`config.yaml`** - Centralized configuration for all modules:

```yaml
system:
  name: "CF-AI-SDE"
  version: "1.0.0"
  log_level: "${LOG_LEVEL}"  # From .env

database:
  mongodb_uri: "${MONGODB_URI}"
  mongodb_database: "${MONGODB_DATABASE}"
  fallback_to_file: true  # Use filesystem if MongoDB unavailable

data_ingestion:
  default_source: "yahoo"
  cache_enabled: true
  cache_ttl_hours: 24

feature_engineering:
  technical_indicators:
    sma_periods: [20, 50, 200]
    ema_periods: [12, 26]
    rsi_period: 14
    # ... 50+ more indicators

ml_models:
  direction:
    model_type: "xgboost"
    train_test_split: 0.8
    validation_split: 0.1
  volatility:
    model_type: "garch"
    # ...

ai_agents:
  enable_sentiment: true
  enable_macro: true
  aggregation_method: "weighted_average"
  # ...

strategies:
  rsi:
    oversold_threshold: 30
    overbought_threshold: 70
  # ...

risk_management:
  max_position_size: 0.20
  var_limit: "${VAR_LIMIT}"
  drawdown_limit: "${DRAWDOWN_LIMIT}"
  # ...

backtesting:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  # ...

rag_mentor:
  chromadb_path: "${CHROMADB_PATH}"
  embedding_model: "${EMBEDDING_MODEL}"
  top_k_results: "${TOP_K_RESULTS}"
  # ...

api_keys:
  gemini: "${GEMINI_API_KEY}"
  groq: "${GROQ_API_KEY}"
  news_api: "${NEWS_API_KEY}"
  # ...
```

### Database Integration

**Hybrid Persistence Strategy**:

1. **MongoDB Primary** (when available):
   - Collections: `market_data_raw`, `market_data_clean`, `market_features`, `agent_outputs`, `agent_memory`, `ml_models`, `backtest_results`, `strategy_performance`, `rag_knowledge`
   - Automatic indexing on `(ticker, timestamp)`, `(model_type, version)`, `(agent_name, timestamp)`, etc.
   - Connection pooling via singleton pattern

2. **Filesystem Fallback** (when MongoDB unavailable):
   - Models saved to `models/` directory (pickle format)
   - Agent weights saved to `agent_weights/` directory (JSON)
   - Backtest results saved to `backtest_results/` directory (JSON)

**Database Setup**:

```bash
# Initialize database (creates collections and indexes)
python setup_database.py
```

**Database Readers/Writers**:

```python
from db.writers import MLModelWriter, AgentMemoryWriter
from db.readers import MLModelReader, AgentMemoryReader

# Save ML model
writer = MLModelWriter()
model_id = writer.save_model(
    model_type='xgboost_direction',
    model_data=pickle.dumps(model),
    metadata={
        'ticker': 'AAPL',
        'accuracy': 0.68,
        'features': feature_names,
        'trained_at': datetime.now()
    }
)

# Load ML model
reader = MLModelReader()
model, metadata = reader.load_model(
    model_type='xgboost_direction',
    version='latest'  # or specific version like 'v3'
)

# Save agent performance
agent_writer = AgentMemoryWriter()
agent_writer.save_agent_weights(
    agent_name='SentimentAgent',
    performance_weight=1.2,  # Increased due to good performance
    accuracy=0.72,
    metadata={'trades': 150, 'profit_factor': 1.8}
)

# Get agent weights
agent_reader = AgentMemoryReader()
weights = agent_reader.get_all_agent_weights()
# Returns: {'SentimentAgent': 1.2, 'MacroAgent': 0.9, ...}
```

---

## 🔧 System Components

### 1. Data Pipeline (`Data-inges-fe/`)

**Purpose**: Ingest, validate, and engineer features from multiple financial data sources.

**Key Modules:**
- **Ingestion**
  - `equity_ohlcv.py` - OHLCV data from Yahoo Finance/Alpha Vantage
  - `news.py` - News sentiment from NewsAPI
  - `macro_data.py` - Economic indicators (GDP, inflation, rates)
  - `vix.py` - Market volatility index

- **Validation**
  - `data_validator.py` - Schema validation, completeness checks, anomaly detection
  - Ensures data quality before feature engineering

- **Feature Engineering**
  - `technical_indicators.py` - 50+ indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
  - `advanced_features.py` - ML features (lags, rolling stats, regime indicators)
  - `interaction_features.py` - Cross-asset correlations, sector rotations

**Output**: Clean, validated data stored in MongoDB with comprehensive features.

---

### 2. ML Models (`ML_Models/`)

**Purpose**: Predictive models for market forecasting and risk assessment.

#### 2.1 Direction Prediction (`direction_pred.py`)
- **Logistic Regression** - Baseline binary classifier
- **Decision Tree** - Rule extraction for interpretability
- **Random Forest** - Feature importance analysis
- **XGBoost** - Gradient boosting for high accuracy
- **Feed Forward NN** - Deep learning with dropout
- **LSTM** - Sequential modeling for time series

#### 2.2 Volatility Forecasting (`Volatility_Forecasting.py`)
- **GARCH** - Baseline econometric model
- **EGARCH** - Asymmetric volatility (leverage effect)
- **LSTM Volatility** - Non-linear deep learning approach

#### 2.3 Regime Classification (`Regime_Classificaiton.py`)
- **Random Forest Classifier** - Traditional ML approach
- **LSTM Classifier** - Deep learning for regime detection
- **Output**: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CRISIS

#### 2.4 Synthetic Data Generation (`GAN.py`)
- **Generative Adversarial Network** - Market simulation
- **Use Cases**: Stress testing, backtesting enhancement

---

### 3. AI Agents (`AI_Agents/`)

**Purpose**: Multi-agent system for distributed market analysis and decision-making.

#### Agent Architecture (`base_agent.py`)
- **BaseAgent** - Abstract class with memory, self-evaluation
- **AgentResponse** - Standardized output (Pydantic schema)
- **Memory System** - Historical analysis storage

#### Specialized Agents (`agents.py`)

| Agent                     | Responsibility                    | Key Outputs                                   |
| ------------------------- | --------------------------------- | --------------------------------------------- |
| **MarketDataAgent**       | OHLCV analysis, anomaly detection | Gap opens, volume spikes, volatility outliers |
| **RiskMonitoringAgent**   | Portfolio risk metrics            | VaR, drawdown, position limits, breach alerts |
| **MacroAgent**            | Economic indicators analysis      | GDP trends, inflation signals, rate impact    |
| **SentimentAgent**        | News sentiment analysis           | Bullish/bearish scores, event detection       |
| **VolatilityAgent**       | Volatility forecasting            | VIX predictions, risk regime changes          |
| **RegimeDetectionAgent**  | Market regime classification      | Current regime, transition probabilities      |
| **SignalAggregatorAgent** | Multi-signal fusion               | Weighted consensus, confidence scores         |

#### Communication (`communication_protocol.py`)
- **Message Router** - Agent-to-agent messaging
- **Priority Queue** - CRITICAL > HIGH > NORMAL > LOW
- **AgentOrchestrator** - Coordinated multi-agent execution

---

### 4. Quantitative Strategy Library (`quant_strategy/`)

**Purpose**: Modular framework for strategy development, backtesting, and risk management.

#### Core Components

**Base Classes (`base.py`)**
- `Action` - Enum: BUY, SELL, HOLD
- `MarketRegime` - Enum: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
- `Context` - Market state container (price, volume, indicators, regime)
- `Signal` - Trading signal with confidence and reasoning
- `BaseStrategy` - Abstract strategy class

**Strategy Engine (`engine.py`)**
- `BacktestEngine` - Full backtesting with temporal integrity
- Features:
  - MongoDB data loading
  - Slippage and commission modeling
  - Portfolio state tracking
  - Trade logging with metadata
  - Performance metrics (Sharpe, Sortino, max DD, win rate)

**Risk Management (`components/risk_manager.py`)**
- **Position Sizing** - Max % of capital per trade
- **Sector Concentration** - Diversification limits
- **Drawdown Limits** - Portfolio heat circuit breakers
- **VaR Monitoring** - Value-at-Risk thresholds
- **Signal Filtering** - Risk-adjusted signal approval

**LangGraph Orchestrator (`components/langgraph_orchestrator.py`)**
- **Dynamic Strategy Graph** - LLM-powered adaptation
- **Nodes**: Data Analysis → Risk Assessment → Signal Generation → Execution
- **State Management** - Stateful multi-turn decision making
- **Integration**: Gemini/Groq LLMs for reasoning

#### Strategy Implementations (`strategies/`)

**1. Technical Strategies**
- **MovingAverageCrossover** - Classic SMA/EMA crossover
- **RSI Mean Reversion** - Oversold/overbought trading
- **MACD Momentum** - Trend-following with MACD
- **Bollinger Band Breakout** - Volatility-based entries
- **Support/Resistance** - Price action levels

**2. ML-Enhanced Strategies**
- **ML Signal Strategy** - Integrates direction prediction models
- **Regime-Adaptive** - Switches strategies based on detected regime
- **Ensemble Strategy** - Combines multiple models with voting

**3. Options Strategies** (`strategies/options.py`)
- **Volatility Arbitrage** - IV vs forecasted realized vol
- **Protective Puts** - Downside hedging
- **Covered Calls** - Premium collection
- **Straddles/Strangles** - Volatility plays

---

### 5. RAG Trading Mentor (`RAG_Mentor/`)

**Purpose**: AI-powered backtest analysis, principle violation detection, and conversational Q&A.

#### Core Technology Stack
- **Vector DB**: ChromaDB with sentence-transformers embeddings
- **LLMs**: Gemini (primary) + Groq (failover)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (free, 384-dim)

#### Knowledge Base
- **20 Trading Principles** from legends (Livermore, Minervini, Buffett, Dalio, etc.)
- **Sample News Articles** with timestamps and symbols
- **Backtest Results Archive** for historical comparison

#### Features

**Performance Analysis (`mentor/performance_analyzer.py`)**
- Correlates backtest results with news events
- Explains performance with market context
- Regime-wise performance breakdown
- Benchmark comparison (vs SPY buy-and-hold)

**Principle Violation Detection (`mentor/principle_checker.py`)**
- **Averaging Down** - Detects adding to losing positions
- **Stop Loss Violations** - Finds losses >10%
- **Overtrading** - Flags holding periods <2 days
- **Position Sizing Issues** - Oversized positions (>20% capital)

**Improvement Suggestions (`mentor/improvement_engine.py`)**
- **LLM-Generated** - Contextual suggestions from principles
- **Programmatic Rules**:
  - Sharpe ratio improvements (volatility-adjusted sizing)
  - Drawdown reduction (heat limits, circuit breakers)
  - Win rate increase (stricter selection, better timing)
  - Regime filters (ADX for ranging, VIX for volatile)
  - Options hedging strategies

**Conversational Interface (`interface/trading_mentor.py`)**
- Intent classification (show_trades, explain, compare, improve)
- Context-aware Q&A
- Trade filtering by date/symbol
- Conversation history and persistence

---

## 🔄 Data Flow

### End-to-End Pipeline

```
1. DATA INGESTION
   └─▶ Yahoo Finance / NewsAPI / FRED
       └─▶ Raw data validation
           └─▶ MongoDB storage (raw collections)

2. FEATURE ENGINEERING
   └─▶ Technical indicators (50+ features)
       └─▶ Advanced features (lags, rolling stats)
           └─▶ MongoDB storage (features collections)

3. ML MODEL TRAINING
   └─▶ Load features from MongoDB
       └─▶ Train models (direction, volatility, regime)
           └─▶ Save models to disk
               └─▶ Store metadata in MongoDB

4. AI AGENT ANALYSIS
   └─▶ Agents query MongoDB for latest data
       └─▶ Generate AgentResponses (market insights)
           └─▶ Message passing via AgentOrchestrator
               └─▶ Consolidated market view

5. STRATEGY EXECUTION
   └─▶ Load data from MongoDB (with temporal constraints)
       └─▶ Generate signals (using strategies + ML models + agents)
           └─▶ Risk Manager validates signals
               └─▶ Execute trades (backtest or live)
                   └─▶ Log trades to MongoDB

6. RAG MENTOR ANALYSIS
   └─▶ Load backtest results
       └─▶ Retrieve relevant principles from ChromaDB
           └─▶ LLM generates analysis
               └─▶ Detect violations + suggest improvements
                   └─▶ Conversational Q&A
```

### Temporal Integrity Guarantees

**Critical**: All modules enforce strict lookahead bias prevention.

- **Data Pipeline**: Timestamps on all data, validation of chronological order
- **Feature Engineering**: Only backward-looking calculations
- **ML Models**: Train/test split respects time ordering
- **Backtesting**: `current_date` parameter ensures only past data is accessible
- **Agent Analysis**: Context restricted to data up to current simulation time

---

## 🛠️ Technology Stack

### Core Libraries

| Category             | Libraries                                                | Purpose                                   |
| -------------------- | -------------------------------------------------------- | ----------------------------------------- |
| **Data Processing**  | `pandas`, `numpy`                                        | Data manipulation and numerical computing |
| **Data Sources**     | `yfinance`, `newsapi-python`, `fredapi`                  | Market data, news, macro indicators       |
| **Visualization**    | `matplotlib`, `seaborn`                                  | Charts and performance visualization      |
| **Machine Learning** | `scikit-learn`, `xgboost`                                | Traditional ML models                     |
| **Deep Learning**    | `tensorflow`, `keras`, `torch`                           | Neural networks (LSTM, GAN)               |
| **Financial Stats**  | `arch`                                                   | GARCH/EGARCH volatility models            |
| **NLP**              | `transformers`, `sentencepiece`, `sentence-transformers` | Sentiment analysis, embeddings            |
| **Database**         | `pymongo`                                                | MongoDB integration                       |
| **LLM APIs**         | `google-generativeai`, `groq`                            | Gemini and Groq LLMs                      |
| **Agent Framework**  | `langchain`, `langgraph`, `langchain-google-genai`       | Multi-agent orchestration                 |
| **Vector DB**        | `chromadb`                                               | Semantic search for RAG                   |
| **Configuration**    | `python-dotenv`, `pydantic`                              | Environment management, data validation   |
| **Utilities**        | `scipy`, `requests`, `coloredlogs`                       | Scientific computing, HTTP, logging       |

### Complete Dependencies

See [requirements.txt](file:///c:/College/Hackathons/Info-Project-AI_CF_SDE/CF-AI-SDE/requirements.txt) for exact versions:
- 56 total dependencies
- All open-source
- Free-tier API support (Gemini, Groq)

---

## ✨ Key Features

### 1. Temporal Integrity
- ✅ Strict lookahead bias prevention
- ✅ Point-in-time data access
- ✅ Chronological validation at every stage

### 2. Multi-Agent Intelligence
- 🤖 7 specialized agents with distinct responsibilities
- 💬 Message-based communication protocol
- 🎯 Coordinated decision-making via orchestrator

### 3. Explainable AI
- 📚 RAG retrieval cites specific trading principles
- 🔍 Violation detection with trade examples
- 💡 LLM-generated natural language explanations

### 4. Regime-Aware Strategies
- 📊 Automatic market regime detection
- 🔄 Dynamic strategy adaptation
- ⚙️ Regime-specific parameter tuning

### 5. Comprehensive Risk Management
- 🛡️ Position size limits
- 📉 Drawdown circuit breakers
- 🎲 VaR monitoring
- 🏢 Sector concentration controls

### 6. Production-Ready
- 💾 MongoDB persistence
- 🔄 Error handling and retries
- 📝 Comprehensive logging
- ⚡ Async agent execution

---

## 🚀 Installation

### Complete Setup Guide

**For detailed step-by-step instructions, see [things_to_get.md](things_to_get.md)**

### Quick Install

```bash
# 1. Clone repository
cd CF-AI-SDE

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and MongoDB URI

# 5. Initialize database
python setup_database.py
```

### System Requirements

- **Python**: 3.10 or 3.11 (3.12 not fully tested)
- **MongoDB**: Local installation or Atlas free tier
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space minimum
- **API Keys** (see things_to_get.md for acquisition):
  - Gemini API (REQUIRED - free tier 60 req/min)
  - Groq API (optional - free tier 30 req/min)
  - NewsAPI (optional - free tier 100 req/day)

---

## ⚙️ Configuration

### Environment Variables (`.env`)

**Required:**
```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=cf_ai_sde

# LLM API (at least one required)
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here  # Optional fallback
```

**Optional but Recommended:**
```env
# News & Sentiment
NEWS_API_KEY=your_news_api_key

# HuggingFace (for FinBERT models)
HUGGINGFACE_TOKEN=your_hf_token

# Risk Management
VAR_LIMIT=0.05           # 5% VaR limit
DRAWDOWN_LIMIT=0.10      # 10% max drawdown

# RAG Mentor
CHROMADB_PATH=./RAG_Mentor/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=5

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### System Configuration (`config.yaml`)

The system uses a centralized YAML configuration file that supports environment variable interpolation:

```yaml
# config.yaml
system:
  name: "CF-AI-SDE"
  version: "1.0.0"
  log_level: "${LOG_LEVEL}"  # Reads from .env

database:
  mongodb_uri: "${MONGODB_URI}"
  mongodb_database: "${MONGODB_DATABASE}"
  fallback_to_file: true  # Use filesystem if MongoDB down

# ... see config.yaml for full configuration options
```

**No need to edit `config.yaml` directly** - all sensitive values are loaded from `.env`.

---

## 🚀 Getting Started

---

## 📁 Project Structure

```
CF-AI-SDE/
├── .env.example                 # Environment template
├── requirements.txt             # All dependencies
├── README.md                    # This file
│
├── Data-inges-fe/              # Data Pipeline
│   ├── main.py                 # Pipeline runner
│   ├── src/
│   │   ├── ingestion/          # Data sources
│   │   ├── validation/         # Data quality
│   │   └── features/           # Feature engineering
│   └── config/                 # Pipeline configs
│
├── ML_Models/                  # Predictive Models
│   ├── direction_pred.py       # Price direction models
│   ├── Volatility_Forecasting.py
│   ├── Regime_Classificaiton.py
│   └── GAN.py                  # Synthetic data
│
├── AI_Agents/                  # Multi-Agent System
│   ├── base_agent.py           # Base classes
│   ├── agents.py               # 7 specialized agents
│   └── communication_protocol.py
│
├── quant_strategy/             # Strategy Framework
│   ├── base.py                 # Core abstractions
│   ├── engine.py               # Backtesting engine
│   ├── components/
│   │   ├── risk_manager.py
│   │   └── langgraph_orchestrator.py
│   ├── strategies/
│   │   ├── technical.py
│   │   ├── ml_enhanced.py
│   │   └── options.py
│   └── example_usage.py
│
└── RAG_Mentor/                 # AI Trading Mentor
    ├── config.py
    ├── vector_db/              # ChromaDB layer
    ├── llm/                    # LLM integration
    ├── knowledge/              # Trading principles
    ├── mentor/                 # Core analysis
    ├── interface/              # User API
    ├── example_usage.py
    ├── README.md
    └── SYSTEM_SUMMARY.md
```

---

## 💡 Usage Examples

### 1. Run Data Pipeline

```python
from Data-inges-fe.main import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(config_path="config/pipeline_config.yaml")

# Ingest data
pipeline.ingest_data(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Engineer features
pipeline.engineer_features()

# Validate and save
pipeline.validate_and_save()
```

### 2. Train ML Models

```python
from ML_Models.direction_pred import Direction_Predictor

# Initialize and train
model = Direction_Predictor(model_type="xgboost")
model.train(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2022-12-31"
)

# Predict
prediction = model.predict(current_features)
```

### 3. Run AI Agents

```python
from AI_Agents.communication_protocol import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# Run all agents
context = {"symbol": "AAPL", "current_date": "2023-06-15"}
results = orchestrator.run_all_agents(context)

# Get signal aggregation
signal = results['SignalAggregatorAgent']
print(f"Recommendation: {signal.recommendation}")
print(f"Confidence: {signal.confidence_score}")
```

### 4. Backtest Strategy

```python
from quant_strategy.engine import BacktestEngine
from quant_strategy.strategies.technical import MovingAverageCrossover

# Initialize strategy
strategy = MovingAverageCrossover(
    fast_period=50,
    slow_period=200
)

# Run backtest
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

results = engine.run(
    strategy=strategy,
    symbol="AAPL",
    start_date="2022-01-01",
    end_date="2023-12-31"
)

print(f"Total Return: {results.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results.metrics['max_drawdown']:.2%}")
```

### 5. Analyze with RAG Mentor

```python
from RAG_Mentor.interface.trading_mentor import TradingMentor

# Initialize mentor
mentor = TradingMentor()

# Analyze backtest
analysis = mentor.analyze_backtest(
    performance_summary={
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.18,
        'win_rate': 0.58
    },
    trades=results.trade_log,
    symbols=['AAPL']
)

# View results
print(analysis['summary'])
print(analysis['violation_report'])
print(analysis['improvement_suggestions'])

# Ask questions
response = mentor.ask_question(
    "Why did the strategy fail in March 2020?"
)
print(response)
```

---

## 📚 Documentation

### Module-Specific Docs

- **RAG Mentor**: [RAG_Mentor/README.md](RAG_Mentor/README.md)
- **Strategy Guide**: [quant_strategy/Strategy_guide.md](quant_strategy/Strategy_guide.md)
- **RAG System Details**: [RAG_Mentor/SYSTEM_SUMMARY.md](RAG_Mentor/SYSTEM_SUMMARY.md)
- **Project Analysis**: [savepoint_3.md](savepoint_3.md)

### Key Design Documents

1. **Architecture Overview** - This README
2. **Data Pipeline Design** - `Data-inges-fe/README.md`
3. **Agent Communication Protocol** - `AI_Agents/communication_protocol.py`
4. **Strategy Development Guide** - `quant_strategy/Strategy_guide.md`
5. **RAG Implementation** - `RAG_Mentor/SYSTEM_SUMMARY.md`

---

## 🎯 Roadmap

### Completed ✅
- [x] Multi-source data ingestion pipeline
- [x] Feature engineering with 50+ indicators
- [x] ML models for direction, volatility, regime
- [x] Multi-agent AI system (7 agents)
- [x] Strategy library with risk management
- [x] Backtesting engine with temporal integrity
- [x] RAG Trading Mentor
- [x] LangGraph orchestration
- [x] MongoDB integration

### In Progress 🚧
- [ ] Live trading connector
- [ ] Real-time data streaming
- [ ] Advanced portfolio optimization
- [ ] Web dashboard (Streamlit/Dash)

### Future Enhancements 🔮
- [ ] Options pricing models
- [ ] Multi-asset portfolio strategies
- [ ] Cloud deployment (AWS/GCP)
- [ ] Real-time alerts (Slack/Email)
- [ ] Backtesting on tick data
- [ ] Reinforcement learning strategies

---

## 🐛 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Errors

**Symptom**: `pymongo.errors.ServerSelectionTimeoutError`

**Solutions**:
```bash
# Check if MongoDB is running
mongosh  # Should connect without errors

# Windows: Start MongoDB service
# Services → MongoDB Server → Start

# macOS: Start MongoDB
brew services start mongodb-community@6.0

# Linux: Start MongoDB
sudo systemctl start mongod

# Verify .env configuration
# MONGODB_URI should be: mongodb://localhost:27017
# Or for Atlas: mongodb+srv://username:password@cluster...
```

#### 2. API Rate Limiting (Gemini/Groq)

**Symptom**: `429 Quota exceeded` or `ResourceExhausted`

**Solutions**:
- **Wait**: Rate limits reset after 1 minute (per-minute limits) or 24 hours (daily limits)
- **Use fallback**: Ensure both `GEMINI_API_KEY` and `GROQ_API_KEY` are set in `.env`
- **Reduce frequency**: Add delays in agent orchestration (see config.yaml `ai_agents.rate_limit_delay`)

#### 3. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'xxx'`

**Solutions**:
```bash
# Verify virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# For specific missing module
pip install <module_name>
```

#### 4. TA-Lib Installation Issues (Windows)

**Symptom**: `ImportError: DLL load failed` or `OSError: [WinError 126]`

**Solutions**:
- Download matching wheel from [TA-Lib Unofficial](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Match Python version (cp311 = Python 3.11) and architecture (amd64 = 64-bit)
- Install: `pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl`
- **Alternative**: System uses `ta` library as fallback (reduced indicator set)

#### 5. ChromaDB Issues

**Symptom**: `chromadb.errors.ChromaError: Could not connect`

**Solutions**:
```bash
# Create ChromaDB directory
mkdir -p RAG_Mentor/chroma_db

# Set permissions (macOS/Linux)
chmod 755 RAG_Mentor/chroma_db

# Reinitialize ChromaDB
python -c "from RAG_Mentor.knowledge.knowledge_loader import KnowledgeLoader; KnowledgeLoader().load_principles()"
```

#### 6. Environment Variables Not Loading

**Symptom**: `KeyError: 'GEMINI_API_KEY'` or similar

**Solutions**:
```bash
# Verify .env file exists in project root
ls -la .env

# Check file contents (redact actual keys when sharing)
cat .env

# Ensure required keys are present:
# - GEMINI_API_KEY
# - MONGODB_URI
# - MONGODB_DATABASE

# Force reload in Python
from dotenv import load_dotenv
load_dotenv(override=True)
```

#### 7. System in Degraded Mode

**Symptom**: `health_check()` returns `status: 'degraded'`

**Cause**: Some modules failed to load or MongoDB unavailable

**Behavior**: System continues with graceful degradation:
- Uses filesystem storage instead of MongoDB
- Skips unavailable agents (e.g., SentimentAgent without NEWS_API_KEY)
- Falls back to simpler models

**Verify**:
```python
api.health_check()
# Check 'modules_loaded' dict for False values
# Check logs for specific error messages
```

### Getting Help

For persistent issues:

1. **Check Logs**: Enable debug logging with `LOG_LEVEL=DEBUG` in `.env`
2. **Search Documentation**: See module-specific READMEs in each directory
3. **Complete Setup Guide**: Review [things_to_get.md](things_to_get.md) for missed steps
4. **GitHub Issues**: Search existing issues or create new one with:
   - Full error traceback
   - Python version (`python --version`)
   - OS version
   - Steps to reproduce
   - Redacted `.env` contents (remove actual API keys!)

---

## 🤝 Contributing

This is a hackathon/research project. Contributions, suggestions, and feedback are welcome!

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

---

*Last Updated: January 2024*
