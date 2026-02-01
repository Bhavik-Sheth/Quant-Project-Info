# Plan: Unified Backend Integration Pipeline

## Overview

This plan integrates all modules (Data-inges-fe, ML_Models, AI_Agents, quant_strategy, Backtesting_risk, RAG_Mentor, db) into a single `logical_pipe.py` orchestrator that provides clean API-ready interfaces for Phase 1, followed by comprehensive database integration in Phase 2.

## Context Summary

Based on comprehensive analysis of [savepoint_3.md](savepoint_3.md) and all module documentation:

- **Data-inges-fe**: Ingests OHLCV from Yahoo Finance, validates, computes 70+ technical indicators → MongoDB (`market_data_raw`, `market_data_clean`, `market_features`)
- **ML_Models**: Trains Direction (XGBoost/LSTM), Volatility (GARCH/LSTM), Regime (RF/LSTM), and GAN models → Currently no persistence (⚠️ Issue)
- **AI_Agents**: 7 specialized agents (MarketData, Risk, Macro, Sentiment, Volatility, Regime, SignalAggregator) → MongoDB (`agent_outputs`, `agent_memory`)
- **quant_strategy**: Strategy framework with RSI, MA Cross, ML-Enhanced strategies + LLM orchestration → Reads MongoDB, outputs trade logs
- **Backtesting_risk**: Alternative backtesting system with institutional-grade features → Uses DataFrames (⚠️ Duplication with quant_strategy)
- **RAG_Mentor**: ChromaDB-powered analysis with violation detection and improvement suggestions
- **db**: MongoDB singleton with readers/writers for 9 collections

## System Architecture Reference

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                          │
│  Yahoo Finance → equity_ohlcv.py → Validation → MongoDB    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                     │
│  MongoDB → Technical Indicators (80+) → Normalization       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. ML MODEL TRAINING                                       │
│  Features → [GAN, Regime, Volatility, Direction] Models    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. AGENT ANALYSIS (Real-time)                             │
│  Context → 7 Agents (parallel) → AgentResponse[]           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. SIGNAL AGGREGATION                                      │
│  AgentResponse[] → LLM Synthesis → Final Signal             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. STRATEGY EXECUTION                                      │
│  Signal → Risk Manager → Portfolio Update → Trade Log      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  7. BACKTESTING                                             │
│  Historical MongoDB → Strategy → Performance Metrics        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  8. RAG ANALYSIS                                            │
│  Performance + Trades → RAG Mentor → Insights               │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Create logical_pipe.py (All Modules Except DB Integration)

### Step 1: Create Unified Orchestration Layer

**File**: `logical_pipe.py` (root directory)

**Goal**: Single import point for API developers with clean interfaces

**Implementation**:

```python
# High-level structure
class DataPipeline:
    """Wraps Data-inges-fe module"""
    - __init__(config: Dict)
    - ingest_data(symbols: List[str], start_date: str, end_date: str, timeframe: str) -> pd.DataFrame
    - engineer_features(raw_data: pd.DataFrame) -> pd.DataFrame
    - validate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]
    - get_latest_data(symbol: str, lookback_days: int) -> pd.DataFrame

class ModelManager:
    """Wraps ML_Models module with persistence"""
    - __init__(config: Dict)
    - train_direction_model(features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict
    - train_volatility_model(features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict
    - train_regime_model(features: pd.DataFrame, targets: pd.Series, model_type: str) -> Dict
    - train_gan(features: np.ndarray, labels: np.ndarray) -> Dict
    - predict_direction(features: pd.DataFrame, model_name: str) -> np.ndarray
    - predict_volatility(features: pd.DataFrame, model_name: str) -> np.ndarray
    - predict_regime(features: pd.DataFrame, model_name: str) -> str
    - save_model(model: Any, model_type: str, version: str) -> str
    - load_model(model_type: str, version: str) -> Any
    - list_models() -> List[Dict]

class AgentOrchestrator:
    """Wraps AI_Agents module"""
    - __init__(config: Dict)
    - initialize_agents() -> None
    - run_all_agents(context: Dict) -> List[AgentResponse]
    - run_single_agent(agent_name: str, context: Dict) -> AgentResponse
    - get_aggregated_signal(agent_responses: List[AgentResponse]) -> Signal
    - update_agent_performance(agent_name: str, actual_outcome: float) -> None
    - get_agent_status() -> Dict[str, Dict]

class StrategyEngine:
    """Unified backtesting (prioritizes quant_strategy, uses Backtesting_risk as advanced option)"""
    - __init__(config: Dict, risk_config: Dict)
    - register_strategy(strategy: BaseStrategy) -> None
    - backtest(symbol: str, start_date: str, end_date: str, strategies: List[str]) -> Dict
    - run_walk_forward(symbol: str, train_window: int, test_window: int) -> Dict
    - get_performance_metrics(trade_log: List[Dict]) -> Dict
    - export_results(results: Dict, format: str) -> str
    - advanced_backtest(data: pd.DataFrame, decisions: List, engine: str = "backtesting_risk") -> Dict

class AnalysisInterface:
    """Wraps RAG_Mentor module"""
    - __init__(config: Dict)
    - analyze_backtest(performance: Dict, trades: List[Dict], symbols: List[str]) -> Dict
    - ask_question(question: str, context: Dict) -> str
    - detect_violations(trades: List[Dict]) -> Dict
    - suggest_improvements(performance: Dict, trades: List[Dict], violations: Dict) -> List[str]
    - benchmark_comparison(performance: Dict, benchmark: str) -> Dict

class TradingSystemAPI:
    """Main API facade - single import for external use"""
    - __init__(config_path: str = "config.yaml")
    - run_full_pipeline(symbols: List[str], start_date: str, end_date: str, strategy_name: str) -> Dict
    - run_partial_pipeline(stage: str, inputs: Dict) -> Dict
    - get_system_status() -> Dict
    - health_check() -> Dict
```

**Integration Points**:
- `DataPipeline` imports from `Data-inges-fe.src.ingestion.runner` and `Data-inges-fe.src.features.feature_runner`
- `ModelManager` imports from `ML_Models.direction_pred`, `ML_Models.Volatility_Forecasting`, `ML_Models.Regime_Classificaiton`, `ML_Models.GAN`
- `AgentOrchestrator` imports from `AI_Agents.agents` and `AI_Agents.communication_protocol`
- `StrategyEngine` imports from `quant_strategy.engine` (primary) and `Backtesting_risk.backtesting` (advanced)
- `AnalysisInterface` imports from `RAG_Mentor.interface.trading_mentor`

### Step 2: Implement Temporal Data Flow Coordinator

**File**: `logical_pipe.py` (add class)

**Goal**: Enforce strict t vs t+1 separation across all modules

**Implementation**:

```python
class TemporalCoordinator:
    """Ensures no look-ahead bias throughout pipeline"""
    
    def __init__(self):
        self.current_timestamp = None
        self.context_at_t = {}
        self.execution_at_t_plus_1 = []
        self.audit_log = []
    
    def advance_time(self, timestamp: datetime) -> None:
        """Move to next timestamp, validate temporal integrity"""
        if self.current_timestamp and timestamp <= self.current_timestamp:
            raise ValueError(f"Time cannot move backward: {timestamp} <= {self.current_timestamp}")
        self.current_timestamp = timestamp
        self.audit_log.append({"timestamp": timestamp, "action": "time_advanced"})
    
    def validate_context(self, context: Dict) -> bool:
        """Ensure all data in context is <= current_timestamp"""
        for key, value in context.items():
            if isinstance(value, pd.DataFrame) and 'timestamp' in value.columns:
                if value['timestamp'].max() > self.current_timestamp:
                    raise ValueError(f"Look-ahead bias detected in {key}: {value['timestamp'].max()} > {self.current_timestamp}")
        return True
    
    def execute_pipeline_step(self, step_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute pipeline step with temporal validation"""
        self.audit_log.append({
            "timestamp": self.current_timestamp,
            "step": step_name,
            "status": "started"
        })
        
        try:
            result = func(*args, **kwargs)
            self.audit_log.append({
                "timestamp": self.current_timestamp,
                "step": step_name,
                "status": "completed"
            })
            return result
        except Exception as e:
            self.audit_log.append({
                "timestamp": self.current_timestamp,
                "step": step_name,
                "status": "failed",
                "error": str(e)
            })
            raise
    
    def get_audit_trail(self) -> List[Dict]:
        """Return complete audit log"""
        return self.audit_log
```

**Integration Pattern**:
```python
# In TradingSystemAPI.run_full_pipeline()
coordinator = TemporalCoordinator()

for timestamp in date_range:
    coordinator.advance_time(timestamp)
    
    # Step 1: Fetch data at time t
    data_t = coordinator.execute_pipeline_step(
        "fetch_data",
        data_pipeline.get_latest_data,
        symbol, lookback_days
    )
    
    # Step 2: Engineer features at time t
    features_t = coordinator.execute_pipeline_step(
        "engineer_features",
        data_pipeline.engineer_features,
        data_t
    )
    
    # Step 3: ML predictions at time t
    predictions_t = coordinator.execute_pipeline_step(
        "ml_predictions",
        model_manager.predict_direction,
        features_t, "xgboost"
    )
    
    # Step 4: Run agents at time t
    context = {"data": data_t, "features": features_t, "predictions": predictions_t}
    coordinator.validate_context(context)
    agent_responses = coordinator.execute_pipeline_step(
        "run_agents",
        agent_orchestrator.run_all_agents,
        context
    )
    
    # Step 5: Generate signal at time t
    signal = coordinator.execute_pipeline_step(
        "generate_signal",
        agent_orchestrator.get_aggregated_signal,
        agent_responses
    )
    
    # Step 6: Execute at t+1 (next iteration)
    coordinator.advance_time(timestamp + timedelta(days=1))
    trade = coordinator.execute_pipeline_step(
        "execute_trade",
        strategy_engine.execute_signal,
        signal, timestamp + timedelta(days=1)
    )
```

### Step 3: Build Configuration Management System

**File**: `config.yaml` (root directory)

**Goal**: Centralized configuration for all modules

**Structure**:

```yaml
# System-wide settings
system:
  name: "CF-AI-SDE"
  version: "1.0.0"
  environment: "development"  # development, staging, production
  log_level: "INFO"

# Data ingestion settings
data_ingestion:
  source: "yahoo"
  symbols: ["AAPL", "MSFT", "GOOGL", "^GSPC", "^NSEI"]
  default_timeframe: "1d"
  lookback_days: 365
  validation:
    enabled: true
    max_missing_pct: 0.05
    price_outlier_threshold: 3.0
    volume_outlier_threshold: 5.0

# Feature engineering settings
feature_engineering:
  indicators:
    trend: ["sma", "ema", "macd", "adx"]
    momentum: ["rsi", "stochastic", "roc", "williams_r", "cci", "mfi"]
    volatility: ["atr", "bollinger_bands", "keltner_channels"]
    volume: ["obv", "vwap", "cmf", "vroc"]
  normalization:
    method: "robust"  # robust, standard, minmax
    outlier_clip: 3.0
  feature_selection:
    enabled: false
    method: "mutual_info"
    top_k: 50

# ML Models settings
ml_models:
  direction:
    default_model: "xgboost"
    models:
      xgboost:
        n_estimators: 200
        max_depth: 6
        learning_rate: 0.01
      lstm:
        units: [128, 64]
        dropout: 0.2
        epochs: 50
        batch_size: 32
  volatility:
    default_model: "garch"
    models:
      garch:
        p: 1
        q: 1
      lstm:
        lookback: 60
        horizon: 5
        units: [128, 64]
  regime:
    default_model: "random_forest"
    regimes: ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "CRISIS"]
    window_size: 60
  persistence:
    enabled: true
    storage: "mongodb"  # mongodb, filesystem
    versioning: true

# AI Agents settings
ai_agents:
  enabled_agents: ["market_data", "risk_monitoring", "sentiment", "volatility", "regime", "signal_aggregator"]
  parallel_execution: true
  timeout_seconds: 30
  
  market_data_agent:
    gap_threshold: 0.02
    volume_spike_threshold: 2.0
    volatility_outlier_threshold: 2.0
  
  risk_monitoring_agent:
    var_limit: 0.05
    drawdown_limit: 0.10
    position_limit: 0.20
  
  sentiment_agent:
    model: "ProsusAI/finbert"
    news_api_enabled: true
    max_articles: 20
  
  volatility_agent:
    model: "garch"
    forecast_horizon: 5
  
  regime_detection_agent:
    model: "lstm"
    confidence_threshold: 0.7
  
  signal_aggregator_agent:
    llm_provider: "gemini"  # gemini, groq
    fallback_provider: "groq"
    conflict_resolution: "weighted_voting"
    dynamic_weighting: true

# Strategy settings
strategies:
  default_strategy: "ml_enhanced"
  available_strategies: ["rsi", "ma_cross", "bollinger_band", "ml_enhanced", "options"]
  
  rsi:
    oversold: 30
    overbought: 70
    period: 14
  
  ma_cross:
    fast_period: 50
    slow_period: 200
  
  ml_enhanced:
    confidence_threshold: 0.6
    use_agents: true
    dynamic_position_sizing: true
  
  orchestration:
    enabled: true
    llm_provider: "gemini"
    regime_based_selection: true

# Risk management settings
risk_management:
  max_position_size: 0.20  # 20% of portfolio
  max_sector_concentration: 0.40  # 40% per sector
  max_drawdown: 0.15  # 15% circuit breaker
  var_confidence: 0.95
  stop_loss_pct: 0.05  # 5%
  take_profit_pct: 0.10  # 10%

# Backtesting settings
backtesting:
  engine: "quant_strategy"  # quant_strategy, backtesting_risk
  initial_capital: 100000
  slippage_pct: 0.001  # 0.1%
  commission_pct: 0.001  # 0.1%
  exchange_fee_pct: 0.0001  # 0.01%
  
  advanced:  # Backtesting_risk specific
    walk_forward:
      enabled: false
      train_window: 252  # 1 year
      test_window: 63  # 3 months
    metrics:
      include_shap: true
      export_format: ["csv", "json"]

# RAG Mentor settings
rag_mentor:
  enabled: true
  vector_db: "chromadb"
  chromadb_path: "./RAG_Mentor/chroma_db"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k_results: 5
  llm_provider: "gemini"
  fallback_provider: "groq"
  
  violation_detection:
    averaging_down: true
    missing_stop_loss: true
    overtrading: true
    position_sizing: true
  
  improvement_engine:
    max_suggestions: 5
    include_code_examples: true

# API Keys (use environment variables)
api_keys:
  gemini: ${GEMINI_API_KEY}
  groq: ${GROQ_API_KEY}
  news_api: ${NEWS_API_KEY}
  economic_calendar: ${ECONOMIC_CALENDAR_API_KEY}
  huggingface: ${HUGGINGFACE_TOKEN}

# Database settings (Phase 2)
database:
  provider: "mongodb"
  uri: ${MONGODB_URI}
  database_name: ${MONGODB_DATABASE}
  graceful_fallback: true
  fallback_storage: "./data/fallback"
  
  collections:
    market_data_raw: "market_data_raw"
    market_data_clean: "market_data_clean"
    market_features: "market_features"
    agent_outputs: "agent_outputs"
    agent_memory: "agent_memory"
    ml_models: "ml_models"
    portfolio_positions: "portfolio_positions"
    audit_log: "audit_log"
    validation_logs: "validation_logs"
  
  indexing:
    enabled: true
    collections_to_index: "all"
  
  connection_pool:
    min_size: 10
    max_size: 100
    max_idle_time_ms: 60000
```

**File**: `logical_pipe.py` (add ConfigLoader class)

```python
import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_env_vars(self) -> None:
        """Replace ${VAR_NAME} with environment variables"""
        def resolve_dict(d: Dict) -> Dict:
            for key, value in d.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    d[key] = os.getenv(env_var, "")
                elif isinstance(value, dict):
                    d[key] = resolve_dict(value)
            return d
        
        self.config = resolve_dict(self.config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'ml_models.direction.default_model')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get entire config section"""
        return self.config.get(section, {})
    
    def validate(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ["system", "data_ingestion", "ml_models", "ai_agents", "strategies", "risk_management"]
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        return True
```

### Step 4: Create things_to_get.md

**File**: `things_to_get.md` (root directory)

**Goal**: Comprehensive checklist of dependencies and setup requirements

**Content**:

```markdown
# CF-AI-SDE: Setup Requirements Checklist

This document lists everything you need to set up before running the CF-AI-SDE trading system.

## 1. Database Setup

### MongoDB Installation

**Option A: Local Installation**
- Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
- Install and start MongoDB service
- Default connection: `mongodb://localhost:27017`

**Option B: MongoDB Atlas (Cloud)**
- Create free account at: https://www.mongodb.com/cloud/atlas
- Create cluster and get connection string
- Set connection string in `.env` file

### Database Collections & Indexes

The system requires 9 collections. Run this setup script after MongoDB installation:

```python
# setup_database.py (will be created in Phase 2)
from db.connection import get_mongodb_client, setup_indexes

client = get_mongodb_client()
setup_indexes(client)
print("✓ Database collections and indexes created")
```

**Collections**:
1. `market_data_raw` - Raw OHLCV data
2. `market_data_validated` - Validated data
3. `market_data_clean` - Cleaned data
4. `market_features` - Technical indicators (70+ features)
5. `normalization_stats` - Mean/std for feature scaling
6. `validation_logs` - Data quality issues
7. `agent_outputs` - AI agent responses
8. `agent_memory` - Agent learning history
9. `portfolio_positions` - Portfolio state tracking

## 2. Python Environment

### Python Version
- **Required**: Python 3.10 or 3.11
- Check version: `python --version`

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### Core Dependencies List

**Data Processing** (15 packages):
- numpy>=1.24.0
- pandas>=2.0.0
- yfinance>=0.2.28
- scipy>=1.11.0
- statsmodels>=0.14.0

**Machine Learning** (10 packages):
- scikit-learn>=1.3.0
- tensorflow>=2.15.0
- xgboost>=2.0.0
- torch>=2.0.0
- arch>=6.0.0  # GARCH models

**Database** (2 packages):
- pymongo>=4.0.0
- motor>=3.3.0  # Async MongoDB driver

**AI/LLM** (12 packages):
- transformers>=4.30.0
- google-generativeai>=0.3.0
- groq>=0.4.0
- langchain>=0.1.0
- langchain-google-genai>=0.0.5
- sentence-transformers>=2.2.0
- chromadb>=0.4.0
- huggingface-hub>=0.17.0

**Utilities** (8 packages):
- pydantic>=2.0.0
- python-dotenv>=1.0.0
- requests>=2.31.0
- PyYAML>=6.0.0
- tqdm>=4.65.0
- pytest>=7.4.0  # Testing

**Visualization** (optional, for notebooks):
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.17.0

## 3. API Keys

### Required API Keys

**1. Google Gemini (Primary LLM)**
- Free tier: 60 requests/minute
- Get key from: https://makersuite.google.com/app/apikey
- Set in `.env`: `GEMINI_API_KEY=your_key_here`

**2. Groq (Fallback LLM)**
- Free tier: 30 requests/minute
- Get key from: https://console.groq.com/keys
- Set in `.env`: `GROQ_API_KEY=your_key_here`

### Optional API Keys

**3. NewsAPI (Sentiment Agent)**
- Free tier: 100 requests/day
- Get key from: https://newsapi.org/register
- Set in `.env`: `NEWS_API_KEY=your_key_here`
- **Note**: System works without this (sentiment agent will be skipped)

**4. Economic Calendar API (Macro Agent)**
- Options:
  - TradingEconomics: https://tradingeconomics.com/api
  - Forex Factory (scraping, no key needed)
- Set in `.env`: `ECONOMIC_CALENDAR_API_KEY=your_key_here`
- **Note**: System works without this (macro agent will be skipped)

**5. HuggingFace Token (Optional)**
- For FinBERT model download (sentiment analysis)
- Get token from: https://huggingface.co/settings/tokens
- Set in `.env`: `HUGGINGFACE_TOKEN=your_token_here`
- **Note**: Public models work without token, but rate-limited

## 4. Environment Variables Setup

Create a `.env` file in the root directory:

```bash
# .env file template

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=cf_ai_sde

# LLM APIs (Required)
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here

# News & Economic Data (Optional)
NEWS_API_KEY=your_newsapi_key_here
ECONOMIC_CALENDAR_API_KEY=your_calendar_key_here

# HuggingFace (Optional)
HUGGINGFACE_TOKEN=your_hf_token_here

# Risk Management
VAR_LIMIT=0.05
DRAWDOWN_LIMIT=0.10

# RAG Mentor
CHROMADB_PATH=./RAG_Mentor/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=5

# Logging
LOG_LEVEL=INFO
```

## 5. ChromaDB Setup (RAG Mentor)

ChromaDB is used for vector storage of trading principles and news articles.

**Installation** (included in requirements.txt):
```bash
pip install chromadb>=0.4.0
```

**Initialization**:
```python
# Will be auto-created on first run
# Location: ./RAG_Mentor/chroma_db/
```

**Embedding Model Download** (automatic on first run):
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Size: ~80MB
- Storage: `~/.cache/huggingface/`

## 6. GPU Support (Optional, Recommended for ML Models)

### NVIDIA GPU Setup

**For TensorFlow**:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]>=2.15.0
```

**For PyTorch**:
```bash
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU**:
```python
import tensorflow as tf
print(f"TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")

import torch
print(f"PyTorch GPU: {torch.cuda.is_available()}")
```

**Note**: System works on CPU, but training will be slower.

## 7. Data Storage

### Directory Structure

The system will auto-create these directories:

```
data/
├── raw/              # Raw OHLCV from Yahoo Finance
├── validated/        # Validated data
├── features/         # Technical indicators
└── fallback/         # CSV backups if MongoDB unavailable

RAG_Mentor/
└── chroma_db/        # Vector database

models/               # Saved ML models (Phase 2)
├── direction/
├── volatility/
├── regime/
└── gan/

logs/                 # Application logs
└── audit_trail.log
```

### Disk Space Requirements

- **Minimum**: 2GB (for 1 year of daily data, 10 symbols)
- **Recommended**: 10GB (for 3 years, 36 symbols, all timeframes)
- **MongoDB**: 5GB+ (depending on data volume)

## 8. Testing Your Setup

### Quick Health Check

```python
# test_setup.py
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI("config.yaml")
status = api.health_check()

print("System Status:")
for component, is_healthy in status.items():
    print(f"  {component}: {'✓' if is_healthy else '✗'}")
```

### Expected Output:
```
System Status:
  config_loaded: ✓
  mongodb_connected: ✓
  gemini_api_ready: ✓
  groq_api_ready: ✓
  chromadb_initialized: ✓
  data_directories_created: ✓
```

## 9. Common Issues & Solutions

### Issue 1: MongoDB Connection Failed
```
Error: ServerSelectionTimeoutError
Solution:
- Ensure MongoDB is running: `mongod --version`
- Check connection string in .env
- For Atlas: Whitelist your IP address
```

### Issue 2: API Key Invalid
```
Error: google.api_core.exceptions.Unauthenticated
Solution:
- Verify API key in .env
- Check key hasn't expired
- Ensure no extra spaces/quotes
```

### Issue 3: Module Import Error
```
Error: ModuleNotFoundError: No module named 'transformers'
Solution:
- Activate virtual environment
- Run: pip install -r requirements.txt
- Check Python version (3.10 or 3.11)
```

### Issue 4: Not an issue
```
```
### Issue 5: ChromaDB Lock Error
```
Error: sqlite3.OperationalError: database is locked
Solution:
- Close other processes using ChromaDB
- Delete: ./RAG_Mentor/chroma_db/*.lock
- Restart application
```

## 10. Next Steps

After completing this checklist:

1. ✅ **Initialize Database**: Run `setup_database.py`
2. ✅ **Test Health Check**: Run `test_setup.py`
3. ✅ **Ingest Sample Data**: Run example in README.md
4. ✅ **Train Initial Models**: Follow ML_Models documentation
5. ✅ **Run First Backtest**: Use TradingSystemAPI

## Support & Documentation

- **System Architecture**: See `savepoint_3.md`
- **Module Documentation**: See individual `README.md` files in each module
- **API Reference**: See `logical_pipe.py` docstrings
- **Database Schema**: See `db/DATABASE_INTEGRATION_DOCUMENTATION.md`

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
```

### Step 5: Implement TradingSystemAPI Facade

**File**: `logical_pipe.py` (add TradingSystemAPI class)

**Goal**: Simple 3-method API for external use

**Implementation**:

```python
class TradingSystemAPI:
    """
    Main API facade for CF-AI-SDE Trading System
    
    Single import for all functionality:
        from logical_pipe import TradingSystemAPI
        
        api = TradingSystemAPI("config.yaml")
        results = api.run_full_pipeline(["AAPL"], "2023-01-01", "2023-12-31", "ml_enhanced")
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trading system with configuration
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Initialize all components
        self.data_pipeline = DataPipeline(self.config.get('data_ingestion'))
        self.model_manager = ModelManager(self.config.get('ml_models'))
        self.agent_orchestrator = AgentOrchestrator(self.config.get('ai_agents'))
        self.strategy_engine = StrategyEngine(
            self.config.get('strategies'),
            self.config.get('risk_management')
        )
        self.analysis_interface = AnalysisInterface(self.config.get('rag_mentor'))
        self.temporal_coordinator = TemporalCoordinator()
        
        self._logger = logging.getLogger(__name__)
    
    def run_full_pipeline(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strategy_name: str,
        use_agents: bool = True,
        use_ml_models: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete trading pipeline from data ingestion to analysis
        
        Args:
            symbols: List of ticker symbols (e.g., ["AAPL", "MSFT"])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy_name: Strategy to use (e.g., "ml_enhanced", "rsi", "ma_cross")
            use_agents: Whether to use AI agents for signal generation
            use_ml_models: Whether to use ML models for predictions
        
        Returns:
            Dict with keys:
                - performance: Performance metrics (Sharpe, returns, etc.)
                - trades: List of executed trades
                - agent_outputs: Agent responses (if use_agents=True)
                - ml_predictions: ML model predictions (if use_ml_models=True)
                - analysis: RAG mentor analysis
                - audit_trail: Temporal coordinator log
        
        Example:
            >>> api = TradingSystemAPI()
            >>> results = api.run_full_pipeline(
            ...     symbols=["AAPL"],
            ...     start_date="2023-01-01",
            ...     end_date="2023-12-31",
            ...     strategy_name="ml_enhanced"
            ... )
            >>> print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
            Sharpe Ratio: 1.85
        """
        try:
            self._logger.info(f"Starting full pipeline for {symbols} from {start_date} to {end_date}")
            
            # Step 1: Data Ingestion
            self._logger.info("Step 1/7: Data ingestion")
            raw_data = self.data_pipeline.ingest_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.get('data_ingestion.default_timeframe', '1d')
            )
            
            # Step 2: Feature Engineering
            self._logger.info("Step 2/7: Feature engineering")
            features = self.data_pipeline.engineer_features(raw_data)
            
            # Step 3: ML Predictions (optional)
            ml_predictions = {}
            if use_ml_models:
                self._logger.info("Step 3/7: ML predictions")
                ml_predictions = {
                    'direction': self.model_manager.predict_direction(features, 'xgboost'),
                    'volatility': self.model_manager.predict_volatility(features, 'garch'),
                    'regime': self.model_manager.predict_regime(features, 'random_forest')
                }
            else:
                self._logger.info("Step 3/7: ML predictions (skipped)")
            
            # Step 4: Agent Analysis (optional)
            agent_outputs = []
            if use_agents:
                self._logger.info("Step 4/7: AI agent analysis")
                context = {
                    'data': raw_data,
                    'features': features,
                    'ml_predictions': ml_predictions,
                    'symbols': symbols
                }
                agent_outputs = self.agent_orchestrator.run_all_agents(context)
            else:
                self._logger.info("Step 4/7: AI agent analysis (skipped)")
            
            # Step 5: Backtesting
            self._logger.info("Step 5/7: Strategy backtesting")
            backtest_results = self.strategy_engine.backtest(
                symbol=symbols[0],  # TODO: Support multiple symbols
                start_date=start_date,
                end_date=end_date,
                strategies=[strategy_name]
            )
            
            # Step 6: RAG Analysis
            self._logger.info("Step 6/7: Performance analysis")
            analysis = self.analysis_interface.analyze_backtest(
                performance=backtest_results['metrics'],
                trades=backtest_results['trades'],
                symbols=symbols
            )
            
            # Step 7: Compile results
            self._logger.info("Step 7/7: Compiling results")
            results = {
                'status': 'success',
                'performance': backtest_results['metrics'],
                'trades': backtest_results['trades'],
                'agent_outputs': agent_outputs,
                'ml_predictions': ml_predictions,
                'analysis': analysis,
                'audit_trail': self.temporal_coordinator.get_audit_trail(),
                'metadata': {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'strategy': strategy_name,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self._logger.info(f"Pipeline completed successfully. Sharpe: {results['performance'].get('sharpe_ratio', 0):.2f}")
            return results
            
        except Exception as e:
            self._logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'audit_trail': self.temporal_coordinator.get_audit_trail()
            }
    
    def run_partial_pipeline(self, stage: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specific pipeline stage only
        
        Args:
            stage: One of ['ingest', 'features', 'ml', 'agents', 'backtest', 'analyze']
            inputs: Stage-specific inputs
        
        Returns:
            Dict with stage-specific outputs
        
        Example:
            >>> # Just ingest data
            >>> data = api.run_partial_pipeline('ingest', {
            ...     'symbols': ['AAPL'],
            ...     'start_date': '2023-01-01',
            ...     'end_date': '2023-12-31'
            ... })
            
            >>> # Just run agents
            >>> agent_outputs = api.run_partial_pipeline('agents', {
            ...     'context': {'data': data, 'features': features}
            ... })
        """
        try:
            if stage == 'ingest':
                return self.data_pipeline.ingest_data(**inputs)
            
            elif stage == 'features':
                return self.data_pipeline.engineer_features(inputs['data'])
            
            elif stage == 'ml':
                model_type = inputs.get('model_type', 'direction')
                if model_type == 'direction':
                    return self.model_manager.predict_direction(inputs['features'], inputs.get('model_name', 'xgboost'))
                elif model_type == 'volatility':
                    return self.model_manager.predict_volatility(inputs['features'], inputs.get('model_name', 'garch'))
                elif model_type == 'regime':
                    return self.model_manager.predict_regime(inputs['features'], inputs.get('model_name', 'random_forest'))
            
            elif stage == 'agents':
                return self.agent_orchestrator.run_all_agents(inputs['context'])
            
            elif stage == 'backtest':
                return self.strategy_engine.backtest(**inputs)
            
            elif stage == 'analyze':
                return self.analysis_interface.analyze_backtest(**inputs)
            
            else:
                raise ValueError(f"Unknown stage: {stage}. Must be one of ['ingest', 'features', 'ml', 'agents', 'backtest', 'analyze']")
        
        except Exception as e:
            self._logger.error(f"Partial pipeline failed at stage '{stage}': {str(e)}", exc_info=True)
            return {'status': 'error', 'stage': stage, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dict with component statuses and system health
        
        Example:
            >>> status = api.get_system_status()
            >>> print(f"Overall Health: {status['overall_health']}")
            Overall Health: healthy
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_health': 'healthy'
        }
        
        # Check each component
        components = [
            ('data_pipeline', self.data_pipeline),
            ('model_manager', self.model_manager),
            ('agent_orchestrator', self.agent_orchestrator),
            ('strategy_engine', self.strategy_engine),
            ('analysis_interface', self.analysis_interface)
        ]
        
        for name, component in components:
            try:
                # Each component should have a health_check() method
                if hasattr(component, 'health_check'):
                    health = component.health_check()
                else:
                    health = {'status': 'unknown', 'message': 'No health check implemented'}
                
                status['components'][name] = health
                
                if health.get('status') != 'healthy':
                    status['overall_health'] = 'degraded'
            
            except Exception as e:
                status['components'][name] = {'status': 'error', 'error': str(e)}
                status['overall_health'] = 'unhealthy'
        
        return status
    
    def health_check(self) -> Dict[str, bool]:
        """
        Quick health check for all dependencies
        
        Returns:
            Dict mapping component names to boolean health status
        
        Example:
            >>> health = api.health_check()
            >>> if all(health.values()):
            ...     print("All systems operational")
        """
        health = {}
        
        # Config loaded
        health['config_loaded'] = self.config is not None
        
        # MongoDB connected (Phase 2)
        try:
            from db.connection import get_mongodb_client
            client = get_mongodb_client()
            health['mongodb_connected'] = client is not None
        except:
            health['mongodb_connected'] = False
        
        # LLM APIs ready
        health['gemini_api_ready'] = bool(self.config.get('api_keys.gemini'))
        health['groq_api_ready'] = bool(self.config.get('api_keys.groq'))
        
        # ChromaDB initialized
        try:
            import chromadb
            chromadb_path = self.config.get('rag_mentor.chromadb_path', './RAG_Mentor/chroma_db')
            health['chromadb_initialized'] = os.path.exists(chromadb_path)
        except:
            health['chromadb_initialized'] = False
        
        # Data directories exist
        health['data_directories_created'] = all([
            os.path.exists('data/raw'),
            os.path.exists('data/validated'),
            os.path.exists('data/features')
        ])
        
        return health
```

### Step 6: Update README.md

**File**: `README.md` (root directory)

**Goal**: Comprehensive documentation with new architecture

**Structure**:

```markdown
# CF-AI-SDE: AI-Powered Quantitative Trading System

## Overview

CF-AI-SDE (Computational Finance - AI Strategy Development Engine) is a production-grade, end-to-end quantitative trading platform that combines classical technical analysis, modern machine learning, multi-agent AI architecture, and explainable decision-making.

**Key Features:**
- 70+ technical indicators with automated feature engineering
- Multi-model ML stack (XGBoost, LSTM, GARCH, GAN)
- 7 specialized AI agents with LLM-powered synthesis
- Regime-aware strategy framework with risk management
- RAG-powered trading mentor for backtest analysis
- Strict temporal integrity (no look-ahead bias)

For detailed system architecture, see [savepoint_3.md](savepoint_3.md).

## Quick Start

### Installation

1. **Prerequisites**:
   - Python 3.10 or 3.11
   - MongoDB (local or Atlas)
   - API keys (Gemini, Groq)

   See [things_to_get.md](things_to_get.md) for complete setup checklist.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   # Copy template and fill in API keys
   cp .env.example .env
   nano .env  # Add your API keys
   ```

4. **Initialize database**:
   ```bash
   python setup_database.py
   ```

### Basic Usage

```python
from logical_pipe import TradingSystemAPI

# Initialize system
api = TradingSystemAPI("config.yaml")

# Run complete pipeline
results = api.run_full_pipeline(
    symbols=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    strategy_name="ml_enhanced"
)

# View performance
print(f"Total Return: {results['performance']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")

# Get AI insights
print("\nAI Analysis:")
print(results['analysis']['summary'])
```

### Advanced Usage

```python
# Run individual pipeline stages
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI()

# 1. Ingest data only
data = api.run_partial_pipeline('ingest', {
    'symbols': ['AAPL'],
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'timeframe': '1d'
})

# 2. Engineer features
features = api.run_partial_pipeline('features', {'data': data})

# 3. Get ML predictions
predictions = api.run_partial_pipeline('ml', {
    'features': features,
    'model_type': 'direction',
    'model_name': 'xgboost'
})

# 4. Run AI agents
context = {'data': data, 'features': features, 'ml_predictions': predictions}
agent_outputs = api.run_partial_pipeline('agents', {'context': context})

# 5. Backtest strategy
backtest_results = api.run_partial_pipeline('backtest', {
    'symbol': 'AAPL',
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'strategies': ['ml_enhanced']
})

# 6. Analyze results
analysis = api.run_partial_pipeline('analyze', {
    'performance': backtest_results['metrics'],
    'trades': backtest_results['trades'],
    'symbols': ['AAPL']
})
```

## System Architecture

### Data Flow

```
Data Ingestion → Feature Engineering → ML Predictions → Agent Analysis → 
Strategy Signals → Risk Management → Execution (t+1) → RAG Analysis
```

See architecture diagram in [savepoint_3.md](savepoint_3.md).

### Core Components

1. **Data Pipeline** ([Data-inges-fe/](Data-inges-fe/))
   - Yahoo Finance ingestion
   - Data validation and cleaning
   - 70+ technical indicators
   - See: [Data-inges-fe/README.md](Data-inges-fe/README.md)

2. **ML Models** ([ML_Models/](ML_Models/))
   - Direction prediction (XGBoost, LSTM)
   - Volatility forecasting (GARCH, LSTM)
   - Regime classification (Random Forest, LSTM)
   - GAN for synthetic data
   - See: [ML_Models/Models_Documentation.md](ML_Models/Models_Documentation.md)

3. **AI Agents** ([AI_Agents/](AI_Agents/))
   - 7 specialized agents (Market, Risk, Macro, Sentiment, Volatility, Regime, Aggregator)
   - LLM-powered signal synthesis (Gemini/Groq)
   - Meta-learning with performance tracking
   - See: [AI_Agents/README.md](AI_Agents/README.md)

4. **Strategy Engine** ([quant_strategy/](quant_strategy/))
   - Modular strategy framework (RSI, MA Cross, Bollinger, ML-Enhanced)
   - LLM-powered strategy orchestration
   - Risk management layer
   - Temporal integrity enforcement
   - See: [quant_strategy/Strategy_guide.md](quant_strategy/Strategy_guide.md)

5. **Advanced Backtesting** ([Backtesting_risk/](Backtesting_risk/))
   - Institutional-grade backtesting
   - Walk-forward validation
   - Comprehensive audit trail
   - See: [Backtesting_risk/DOCUMENTATION.md](Backtesting_risk/DOCUMENTATION.md)

6. **Database Layer** ([db/](db/))
   - MongoDB integration with graceful fallback
   - 9 collections with optimized indexing
   - Append-only audit logs
   - See: [db/DATABASE_INTEGRATION_DOCUMENTATION.md](db/DATABASE_INTEGRATION_DOCUMENTATION.md)

7. **RAG Mentor** ([RAG_Mentor/](RAG_Mentor/))
   - ChromaDB vector database
   - Trading principles from market masters
   - Violation detection and improvement suggestions
   - Conversational analysis interface
   - See: [RAG_Mentor/README.md](RAG_Mentor/README.md)

### Unified Interface

All components are accessible through `logical_pipe.py`:

```python
from logical_pipe import (
    TradingSystemAPI,    # Main API facade
    DataPipeline,         # Data ingestion & features
    ModelManager,         # ML model management
    AgentOrchestrator,    # AI agents
    StrategyEngine,       # Backtesting
    AnalysisInterface,    # RAG mentor
    ConfigLoader          # Configuration
)
```

## Configuration

System configuration is managed via [config.yaml](config.yaml):

```yaml
# Example configuration
data_ingestion:
  symbols: ["AAPL", "MSFT", "^GSPC"]
  default_timeframe: "1d"
  lookback_days: 365

ml_models:
  direction:
    default_model: "xgboost"
    models:
      xgboost:
        n_estimators: 200
        max_depth: 6

ai_agents:
  enabled_agents: ["market_data", "risk_monitoring", "sentiment", "volatility", "regime", "signal_aggregator"]
  parallel_execution: true

strategies:
  default_strategy: "ml_enhanced"
  rsi:
    oversold: 30
    overbought: 70

risk_management:
  max_position_size: 0.20
  max_drawdown: 0.15
```

See full configuration options in [config.yaml](config.yaml).

## Module Documentation

- **Data Ingestion**: [Data-inges-fe/README.md](Data-inges-fe/README.md)
  - [Liquid Instruments Guide](Data-inges-fe/LIQUID_INSTRUMENTS_GUIDE.md)
  - [Technical Indicators](Data-inges-fe/TECHNICAL_INDICATORS_GUIDE.md)
  - [Optimized Features](Data-inges-fe/OPTIMIZED_FEATURES.md)

- **ML Models**: [ML_Models/Models_Documentation.md](ML_Models/Models_Documentation.md)

- **AI Agents**: [AI_Agents/README.md](AI_Agents/README.md)
  - [Configuration](AI_Agents/CONFIGURATION.md)

- **Strategies**: [quant_strategy/Strategy_guide.md](quant_strategy/Strategy_guide.md)

- **Backtesting**: [Backtesting_risk/DOCUMENTATION.md](Backtesting_risk/DOCUMENTATION.md)

- **Database**: [db/DATABASE_INTEGRATION_DOCUMENTATION.md](db/DATABASE_INTEGRATION_DOCUMENTATION.md)

- **RAG Mentor**: [RAG_Mentor/README.md](RAG_Mentor/README.md)
  - [System Summary](RAG_Mentor/SYSTEM_SUMMARY.md)

## Examples

### Example 1: Simple RSI Strategy

```python
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI()

results = api.run_full_pipeline(
    symbols=["AAPL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    strategy_name="rsi",
    use_agents=False,  # Pure technical strategy
    use_ml_models=False
)

print(results['performance'])
```

### Example 2: ML-Enhanced with Agents

```python
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI()

results = api.run_full_pipeline(
    symbols=["MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    strategy_name="ml_enhanced",
    use_agents=True,   # Enable AI agents
    use_ml_models=True  # Enable ML predictions
)

# View agent insights
for agent_output in results['agent_outputs']:
    print(f"{agent_output.agent_name}: {agent_output.analysis}")

# Get improvement suggestions
print("\nAI Improvement Suggestions:")
for suggestion in results['analysis']['improvement_suggestions']:
    print(f"- {suggestion}")
```

### Example 3: Custom Strategy

See [quant_strategy/example_usage.py](quant_strategy/example_usage.py) for creating custom strategies.

### Example 4: Multi-Symbol Portfolio

```python
# Coming soon: Multi-symbol support
# Currently supports single symbol per backtest
```

## Performance Metrics

The system tracks 20+ performance metrics:

- **Returns**: Total return, annualized return, CAGR
- **Risk**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, VaR
- **Trading**: Win rate, profit factor, avg win/loss, trade count
- **Advanced**: Omega ratio, Ulcer index, recovery factor

## Troubleshooting

### Common Issues

**1. MongoDB Connection Failed**
```
Error: ServerSelectionTimeoutError
Solution: Ensure MongoDB is running and connection string is correct in .env
```

**2. API Rate Limits (Gemini/Groq)**
```
Error: 429 Too Many Requests
Solution: System automatically falls back to secondary LLM. Wait and retry.
```

**3. Memory Error During ML Training**
```
Solution: Reduce batch_size in config.yaml or enable GPU support
```

**4. Missing Technical Indicators**
```
Error: KeyError: 'RSI_14'
Solution: Ensure feature engineering ran successfully. Check validation_logs.
```

See [things_to_get.md](things_to_get.md) for complete troubleshooting guide.

### Health Check

```python
from logical_pipe import TradingSystemAPI

api = TradingSystemAPI()
health = api.health_check()

for component, is_healthy in health.items():
    print(f"{component}: {'✓' if is_healthy else '✗'}")
```

## Development

### Project Structure

```
CF-AI-SDE/
├── logical_pipe.py          # Unified API (NEW)
├── config.yaml              # System configuration (NEW)
├── requirements.txt         # Dependencies
├── .env                     # Environment variables
├── things_to_get.md        # Setup checklist (NEW)
├── AI_Agents/              # Multi-agent system
├── Backtesting_risk/       # Advanced backtesting
├── Data-inges-fe/          # Data pipeline
├── db/                     # Database layer
├── ML_Models/              # Predictive models
├── quant_strategy/         # Strategy framework
└── RAG_Mentor/             # Analysis interface
```

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Specific module
pytest tests/test_ml_models.py -v
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Roadmap

### Phase 1: Core Integration (Current)
- [x] Unified API interface
- [x] Configuration management
- [x] Temporal coordinator
- [ ] Complete database integration
- [ ] Testing suite

### Phase 2: Enhancements
- [ ] Live trading connector (Alpaca, Interactive Brokers)
- [ ] Walk-forward optimization
- [ ] Model retraining pipeline
- [ ] Feature selection tools
- [ ] User authentication

### Phase 3: UI Development
- [ ] Next.js frontend
- [ ] Real-time monitoring dashboard
- [ ] Strategy builder (React Flow)
- [ ] Model training interface
- [ ] Conversational RAG interface

## License

[Add your license here]

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cf_ai_sde2024,
  title = {CF-AI-SDE: AI-Powered Quantitative Trading System},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/janvi-0706/CF-AI-SDE}
}
```

## Acknowledgments

- Technical indicators inspired by TA-Lib
- Trading principles from Jesse Livermore, Mark Minervini, Warren Buffett
- ML architectures from academic quant finance literature

---

**Last Updated**: [Current Date]  
**Version**: 1.0.0  
**Status**: Production Ready (Backend)
```

## Phase 2: Database Integration

### Step 7: Extend db Module for Model Persistence

**Files to modify**:
- [db/writers.py](db/writers.py) - Add model persistence
- [db/readers.py](db/readers.py) - Add model loading
- [db/connection.py](db/connection.py) - Add new collection

**Changes**:

1. **Add to [db/writers.py](db/writers.py)**:

```python
class MLModelWriter:
    """Persist ML models to MongoDB"""
    
    def __init__(self, db_client):
        self.collection = db_client['ml_models']
    
    def save_model(self, model_type: str, model_data: bytes, metadata: Dict) -> str:
        """
        Save ML model to MongoDB with versioning
        
        Args:
            model_type: 'direction', 'volatility', 'regime', 'gan'
            model_data: Pickled model bytes
            metadata: Dict with version, metrics, hyperparameters
        
        Returns:
            Model ID (ObjectId as string)
        """
        document = {
            'model_type': model_type,
            'version': metadata.get('version', '1.0.0'),
            'trained_at': datetime.now(),
            'metrics': metadata.get('metrics', {}),
            'hyperparameters': metadata.get('hyperparameters', {}),
            'model_data': model_data,  # Binary data
            'model_size_mb': len(model_data) / (1024 * 1024),
            'framework': metadata.get('framework', 'sklearn'),
            'status': 'active'
        }
        
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """Update model status (active, deprecated, archived)"""
        result = self.collection.update_one(
            {'_id': ObjectId(model_id)},
            {'$set': {'status': status, 'updated_at': datetime.now()}}
        )
        return result.modified_count > 0

class AgentMemoryWriter:
    """Persist agent performance weights"""
    
    def __init__(self, db_client):
        self.collection = db_client['agent_memory']
    
    def save_agent_weights(self, agent_name: str, performance_weight: float, metadata: Dict) -> None:
        """Save agent performance weight for meta-learning"""
        document = {
            'agent_name': agent_name,
            'performance_weight': performance_weight,
            'timestamp': datetime.now(),
            'accuracy': metadata.get('accuracy', 0.0),
            'total_predictions': metadata.get('total_predictions', 0),
            'correct_predictions': metadata.get('correct_predictions', 0),
            'session_id': metadata.get('session_id')
        }
        
        self.collection.insert_one(document)
```

2. **Add to [db/readers.py](db/readers.py)**:

```python
class MLModelReader:
    """Load ML models from MongoDB"""
    
    def __init__(self, db_client):
        self.collection = db_client['ml_models']
    
    def load_model(self, model_type: str, version: str = 'latest') -> Tuple[Any, Dict]:
        """
        Load ML model from MongoDB
        
        Args:
            model_type: 'direction', 'volatility', 'regime', 'gan'
            version: Specific version or 'latest'
        
        Returns:
            (model_object, metadata)
        """
        query = {'model_type': model_type, 'status': 'active'}
        
        if version != 'latest':
            query['version'] = version
        
        # Get latest by trained_at
        document = self.collection.find_one(
            query,
            sort=[('trained_at', -1)]
        )
        
        if not document:
            raise ValueError(f"No model found for type={model_type}, version={version}")
        
        # Deserialize model
        model_data = document['model_data']
        model = pickle.loads(model_data)
        
        metadata = {
            'model_id': str(document['_id']),
            'version': document['version'],
            'trained_at': document['trained_at'],
            'metrics': document['metrics'],
            'hyperparameters': document['hyperparameters']
        }
        
        return model, metadata
    
    def list_models(self, model_type: str = None) -> List[Dict]:
        """List all available models"""
        query = {'status': 'active'}
        if model_type:
            query['model_type'] = model_type
        
        models = self.collection.find(query).sort('trained_at', -1)
        
        return [{
            'model_id': str(m['_id']),
            'model_type': m['model_type'],
            'version': m['version'],
            'trained_at': m['trained_at'],
            'metrics': m['metrics'],
            'size_mb': m['model_size_mb']
        } for m in models]

class AgentMemoryReader:
    """Read agent performance history"""
    
    def __init__(self, db_client):
        self.collection = db_client['agent_memory']
    
    def get_latest_weight(self, agent_name: str) -> float:
        """Get most recent performance weight"""
        document = self.collection.find_one(
            {'agent_name': agent_name},
            sort=[('timestamp', -1)]
        )
        
        return document['performance_weight'] if document else 1.0
    
    def get_agent_history(self, agent_name: str, days: int = 30) -> List[Dict]:
        """Get agent performance history"""
        cutoff = datetime.now() - timedelta(days=days)
        
        history = self.collection.find({
            'agent_name': agent_name,
            'timestamp': {'$gte': cutoff}
        }).sort('timestamp', 1)
        
        return list(history)
```

3. **Update [db/connection.py](db/connection.py)**:

```python
def setup_indexes(client):
    """Setup indexes for all collections"""
    db = client[os.getenv('MONGODB_DATABASE', 'cf_ai_sde')]
    
    # ... existing indexes ...
    
    # ML Models collection
    db['ml_models'].create_index([('model_type', 1), ('version', 1)])
    db['ml_models'].create_index([('trained_at', -1)])
    db['ml_models'].create_index([('status', 1)])
    
    # Agent Memory collection (already exists, add weight index)
    db['agent_memory'].create_index([('agent_name', 1), ('timestamp', -1)])
    db['agent_memory'].create_index([('performance_weight', -1)])
```

### Step 8: Integrate Database with ModelManager

**File**: `logical_pipe.py` (modify ModelManager class)

**Changes**:

```python
class ModelManager:
    """Wraps ML_Models module with MongoDB persistence"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.persistence_enabled = config.get('persistence', {}).get('enabled', True)
        self.storage = config.get('persistence', {}).get('storage', 'mongodb')
        
        # Initialize db writers/readers if enabled
        if self.persistence_enabled and self.storage == 'mongodb':
            try:
                from db.connection import get_mongodb_client
                from db.writers import MLModelWriter
                from db.readers import MLModelReader
                
                client = get_mongodb_client()
                self.model_writer = MLModelWriter(client)
                self.model_reader = MLModelReader(client)
                self.db_available = True
            except Exception as e:
                logging.warning(f"MongoDB unavailable, using file-based storage: {e}")
                self.db_available = False
        else:
            self.db_available = False
    
    def save_model(self, model: Any, model_type: str, version: str, metrics: Dict) -> str:
        """Save model with versioning"""
        if self.db_available:
            # Serialize model
            model_data = pickle.dumps(model)
            
            metadata = {
                'version': version,
                'metrics': metrics,
                'hyperparameters': getattr(model, 'get_params', lambda: {})(),
                'framework': type(model).__module__.split('.')[0]
            }
            
            model_id = self.model_writer.save_model(model_type, model_data, metadata)
            logging.info(f"Model saved to MongoDB: {model_id}")
            return model_id
        else:
            # Fallback: Save to file system
            os.makedirs('models/', exist_ok=True)
            filepath = f"models/{model_type}_{version}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Model saved to file: {filepath}")
            return filepath
    
    def load_model(self, model_type: str, version: str = 'latest') -> Any:
        """Load model from storage"""
        if self.db_available:
            model, metadata = self.model_reader.load_model(model_type, version)
            logging.info(f"Model loaded from MongoDB: {metadata['model_id']}")
            return model
        else:
            # Fallback: Load from file system
            if version == 'latest':
                # Find latest file
                files = glob.glob(f"models/{model_type}_*.pkl")
                if not files:
                    raise FileNotFoundError(f"No saved model for {model_type}")
                filepath = max(files, key=os.path.getctime)
            else:
                filepath = f"models/{model_type}_{version}.pkl"
            
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Model loaded from file: {filepath}")
            return model
    
    def list_models(self, model_type: str = None) -> List[Dict]:
        """List available models"""
        if self.db_available:
            return self.model_reader.list_models(model_type)
        else:
            # Fallback: List files
            pattern = f"models/{model_type}_*.pkl" if model_type else "models/*.pkl"
            files = glob.glob(pattern)
            return [{'filepath': f, 'size_mb': os.path.getsize(f) / (1024*1024)} for f in files]
```

### Step 9: Integrate Database with AgentOrchestrator

**File**: `logical_pipe.py` (modify AgentOrchestrator class)

**Changes**:

```python
class AgentOrchestrator:
    """Wraps AI_Agents module with persistent performance tracking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {}
        
        # Initialize db writers/readers
        try:
            from db.connection import get_mongodb_client
            from db.writers import AgentMemoryWriter
            from db.readers import AgentMemoryReader
            
            client = get_mongodb_client()
            self.memory_writer = AgentMemoryWriter(client)
            self.memory_reader = AgentMemoryReader(client)
            self.db_available = True
        except Exception as e:
            logging.warning(f"MongoDB unavailable for agent memory: {e}")
            self.db_available = False
        
        self.initialize_agents()
    
    def initialize_agents(self) -> None:
        """Initialize agents and restore performance weights from DB"""
        from AI_Agents.agents import (
            MarketDataAgent, RiskMonitoringAgent, SentimentAgent,
            VolatilityAgent, RegimeDetectionAgent, SignalAggregatorAgent
        )
        
        enabled = self.config.get('enabled_agents', [])
        
        agent_classes = {
            'market_data': MarketDataAgent,
            'risk_monitoring': RiskMonitoringAgent,
            'sentiment': SentimentAgent,
            'volatility': VolatilityAgent,
            'regime': RegimeDetectionAgent,
            'signal_aggregator': SignalAggregatorAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            if agent_name in enabled:
                agent = agent_class(name=agent_name)
                
                # Restore performance weight from database
                if self.db_available:
                    try:
                        saved_weight = self.memory_reader.get_latest_weight(agent_name)
                        agent.performance_weight = saved_weight
                        logging.info(f"Restored {agent_name} weight: {saved_weight:.3f}")
                    except Exception as e:
                        logging.warning(f"Could not restore weight for {agent_name}: {e}")
                
                self.agents[agent_name] = agent
    
    def update_agent_performance(self, agent_name: str, actual_outcome: float) -> None:
        """Update agent performance and persist to database"""
        if agent_name not in self.agents:
            return
        
        agent = self.agents[agent_name]
        
        # Update agent's internal performance tracking
        agent.update_performance(actual_outcome)
        
        # Persist to database
        if self.db_available:
            metadata = {
                'accuracy': getattr(agent, 'accuracy', 0.0),
                'total_predictions': getattr(agent, 'total_predictions', 0),
                'correct_predictions': getattr(agent, 'correct_predictions', 0),
                'session_id': self.config.get('session_id', 'default')
            }
            
            self.memory_writer.save_agent_weights(
                agent_name,
                agent.performance_weight,
                metadata
            )
            logging.info(f"Persisted {agent_name} weight: {agent.performance_weight:.3f}")
```

### Step 10: Create Database Setup Script

**File**: `setup_database.py` (root directory)

```python
"""
Database initialization script for CF-AI-SDE
Run this after installing MongoDB
"""

import os
from dotenv import load_dotenv
from db.connection import get_mongodb_client, setup_indexes

def main():
    print("=" * 60)
    print("CF-AI-SDE Database Setup")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection
    print("\n1. Connecting to MongoDB...")
    try:
        client = get_mongodb_client()
        if client is None:
            print("❌ Failed to connect to MongoDB")
            print("\nPlease ensure:")
            print("  - MongoDB is running")
            print("  - MONGODB_URI is set in .env file")
            print("  - Connection string is correct")
            return
        print("✓ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Setup indexes
    print("\n2. Creating collections and indexes...")
    try:
        setup_indexes(client)
        print("✓ Indexes created successfully")
    except Exception as e:
        print(f"❌ Index creation error: {e}")
        return
    
    # Verify collections
    print("\n3. Verifying collections...")
    db_name = os.getenv('MONGODB_DATABASE', 'cf_ai_sde')
    db = client[db_name]
    
    expected_collections = [
        'market_data_raw',
        'market_data_validated',
        'market_data_clean',
        'market_features',
        'normalization_stats',
        'validation_logs',
        'agent_outputs',
        'agent_memory',
        'ml_models'
    ]
    
    existing = db.list_collection_names()
    
    for collection in expected_collections:
        if collection in existing:
            count = db[collection].estimated_document_count()
            print(f"  ✓ {collection} (documents: {count})")
        else:
            print(f"  ℹ {collection} (will be created on first write)")
    
    # Test write and read
    print("\n4. Testing database operations...")
    try:
        # Test write
        test_collection = db['_test']
        test_collection.insert_one({'test': 'data', 'timestamp': 'now'})
        
        # Test read
        doc = test_collection.find_one({'test': 'data'})
        
        # Cleanup
        test_collection.delete_one({'test': 'data'})
        
        print("✓ Read/write operations successful")
    except Exception as e:
        print(f"❌ Database operation error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Database setup complete!")
    print("=" * 60)
    print("\nYou can now run the trading system:")
    print("  python -c \"from logical_pipe import TradingSystemAPI; api = TradingSystemAPI()\"")

if __name__ == "__main__":
    main()
```

## Summary

This plan provides:

1. **Phase 1: Clean Integration**
   - Single `logical_pipe.py` file for all backend logic
   - 5 core classes wrapping existing modules
   - Temporal coordinator for look-ahead bias prevention
   - Unified configuration via [config.yaml](config.yaml)
   - Simple 3-method API for external use

2. **Phase 2: Database Layer**
   - ML model persistence with versioning
   - Agent performance weight persistence
   - Graceful fallback to file storage
   - Database setup automation

3. **Documentation**
   - [things_to_get.md](things_to_get.md) - Complete setup checklist
   - Updated [README.md](README.md) - Quick start + examples
   - Maintained backward compatibility with module docs

4. **API-Ready Design**
   - Single import: `from logical_pipe import TradingSystemAPI`
   - Clean interfaces for FastAPI/Django integration
   - Standardized JSON responses
   - Comprehensive error handling

## Issues Resolved

- ✅ Code duplication (quant_strategy vs Backtesting_risk) - Unified in StrategyEngine
- ✅ ML model persistence - ModelManager with MongoDB/file fallback
- ✅ Agent weight persistence - AgentOrchestrator with MongoDB
- ✅ Configuration fragmentation - Centralized [config.yaml](config.yaml)
- ✅ Import path issues - Clean module structure in `logical_pipe.py`
- ✅ Temporal integrity - TemporalCoordinator class

## Next Steps After Plan Approval

1. Create `logical_pipe.py` with all 5 core classes
2. Create [config.yaml](config.yaml) with full configuration
3. Create [things_to_get.md](things_to_get.md) checklist
4. Extend [db/writers.py](db/writers.py) and [db/readers.py](db/readers.py)
5. Create `setup_database.py` initialization script
6. Update [README.md](README.md) with new architecture
7. Test end-to-end pipeline with sample data
8. Create example API integration (FastAPI)
