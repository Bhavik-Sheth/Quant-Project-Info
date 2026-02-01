# CF-AI-SDE: Comprehensive AI-Powered Quantitative Trading System

## System Architecture

The codebase is organized into **4 major modules** that work together in a pipeline.

---

## 1. Data-inges-fe Module (Data Ingestion & Feature Engineering)

**Purpose**: Fetch, validate, and transform market data into ML-ready features

**Data Flow**:
```
Yahoo Finance API → Ingestion → Validation → Feature Engineering → MongoDB
```

### Features:

#### Data Ingestion (`equity_ohlcv.py`)
- Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Supports multiple timeframes (1m, 5m, 1h, 1d)
- Handles both raw and adjusted prices (splits/dividends)
- UTC timezone normalization
  
#### Data Validation (`ohlcv_checks.py`)
- Temporal integrity checks (no future data leakage)
- Price sanity checks
- Volume anomaly detection
- Missing data handling

#### Technical Indicators (`technical_indicators.py`)
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, ROC, Stochastic Oscillator, Williams %R, CCI, MFI, Ultimate Oscillator
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Volume**: OBV, VWAP, Chaikin Money Flow, Volume Rate of Change
- **Pattern Recognition**: Candlestick patterns, Support/Resistance detection

#### Normalization (`normalization.py`)
- Feature scaling for ML models
- Handling of outliers

---

## 2. ML_Models Module (Predictive Models)

**Purpose**: Train ML models for market prediction tasks

### Model Components:

#### A. GAN.py - Generative Adversarial Network
- Generates synthetic market scenarios for stress testing
- Conditional GAN based on market regimes
- Creates realistic price sequence distributions

#### B. Regime_Classification.py
- **Random Forest Classifier**: Instant regime classification
- **LSTM Classifier**: Sequence-based regime detection with transition probabilities
- Detects: Range, Trending Up, Trending Down, Crisis
- Output: Regime predictions with confidence scores

#### C. Volatility_Forecasting.py
- **Baseline**: Linear Regression
- **Non-linear**: Random Forest & XGBoost regressors
- **Time Series**: GARCH, EGARCH models for volatility clustering
- **Deep Learning**: LSTM for multi-day volatility forecasting
- Output: Predicted volatility levels

#### D. direction_pred.py - Price Direction Prediction
- **Baseline**: Logistic Regression
- **Rule Extraction**: Decision Tree for interpretable rules
- **Feature Importance**: Random Forest rankings
- **Advanced**: XGBoost, Feed-Forward NN, LSTM
- Output: Buy/Sell direction with probability

---

## 3. AI_Agents Module (Multi-Agent Intelligence System)

**Purpose**: Autonomous agents that analyze market from different perspectives

**Agent Architecture**:
```
BaseAgent (Abstract) → 7 Specialized Agents → SignalAggregatorAgent (LLM Synthesis)
```

### 7 Specialized Agents:

#### Deterministic Agents:

**1. MarketDataAgent**
- Fetches OHLCV from MongoDB
- Detects anomalies (gap opens, volume spikes, volatility outliers)
- Statistical threshold-based alerts

**2. RiskMonitoringAgent**
- Calculates Value at Risk (VaR) using historical simulation
- Monitors drawdown limits
- Circuit breaker triggers
- Risk limit breach detection

**3. MacroAgent**
- Fetches upcoming economic events from API
- Analyzes historical market reactions
- Event impact forecasting (Fed meetings, CPI, GDP)

#### Probabilistic Agents (ML-powered):

**4. SentimentAgent**
- Uses FinBERT transformer for news sentiment
- Fetches real-time news via News API
- Aggregate sentiment scoring
- Momentum detection (improving/deteriorating)

**5. VolatilityAgent**
- GARCH/EGARCH volatility forecasting
- Regime change detection (expanding/compressing)
- Implied vs Realized volatility comparison

**6. RegimeDetectionAgent**
- Uses trained Regime Classifier
- HMM-based regime transitions
- Strategy recommendations per regime

#### Synthesis Agent:

**7. SignalAggregatorAgent**
- Aggregates all agent outputs
- Uses LLM (Gemini/Groq) for reasoning
- Conflict resolution logic
- Dynamic agent weighting based on performance
- Final BUY/SELL/HOLD decision

### Key Innovation: Meta-Learning
- Each agent stores memory of predictions vs actual outcomes
- Automatically adjusts `performance_weight` based on accuracy
- SignalAggregator weights agents dynamically

### Communication Protocol (`communication_protocol.py`)
- Message routing between agents
- Priority-based message queues
- AgentOrchestrator for workflow coordination
- Serialization/deserialization

---

## 4. quant_strategy Module (Trading Strategies)

**Purpose**: Modular strategy library with backtesting engine

### Base Infrastructure (`base.py`)
- **Enums**: Action (BUY/SELL/HOLD), Regime, StrategyType
- **Context**: Immutable market state snapshot at time t (prevents look-ahead bias)
- **Signal**: Explainable signal with confidence and natural language reason
- **PortfolioState**: Cash, positions, equity curve, drawdown tracking

### Strategy Implementations:

**1. TechnicalStrategy** (`technical.py`)
- Moving average crossovers
- RSI overbought/oversold
- Bollinger Band mean reversion
- Multi-indicator fusion

**2. MLEnhancedStrategy** (`ml_enhanced.py`)
- Integrates ML_Models predictions
- Direction prediction + volatility forecast
- Dynamic position sizing based on confidence
- Regime-aware parameter adjustment

**3. OptionsStrategy** (`options.py`)
- Greeks-based hedging
- Volatility arbitrage
- Covered call strategies

### Risk Management (`risk_manager.py`)
- Position size limits (% of portfolio)
- Sector concentration checks
- Drawdown circuit breakers
- Value at Risk (VaR) monitoring
- Signal validation gates

### Ensemble (`ensemble.py`)
- Combines multiple strategies
- Weighted voting
- Conflict resolution

### Backtesting Engine (`engine.py`)

**Critical Innovation**: Strict temporal integrity

```
Flow: Fetch data at t → Build context at t → Generate signal at t → Execute trade at t+1
```

- MongoDB batch fetching (memory efficient)
- ML model integration
- Performance metrics (Sharpe, drawdown, win rate)
- Trade log export

---

## Complete Data Flow

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
```

---

## Key Features & Innovations

1. **White Box AI**: Every signal includes natural language reasoning
2. **Temporal Integrity**: Strict t vs t+1 separation prevents look-ahead bias
3. **Meta-Learning**: Agents learn from mistakes and self-adjust weights
4. **Multi-Agent Collaboration**: 7 specialized agents with LLM orchestration
5. **Explainability**: Decision trees extract human-readable rules
6. **Regime Awareness**: Strategies adapt to market conditions
7. **Comprehensive Risk Management**: VaR, drawdown, position limits
8. **Production-Ready**: MongoDB integration, environment configs, error handling

---

## Supported Markets & Instruments

- Equities (via Yahoo Finance)
- Indices (NSE, BSE)
- Options (Greeks-based strategies)
- Multiple timeframes (1min to 1day)

---

## System Characteristics

This is a **sophisticated, production-grade quantitative trading system** combining:
- Classical technical analysis
- Modern machine learning
- Multi-agent AI architecture
- Rigorous risk management
- Explainable decision-making
