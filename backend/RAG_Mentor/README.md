# RAG Trading Mentor 🤖📈

A sophisticated RAG (Retrieval-Augmented Generation) system that acts as an intelligent Trading Mentor, providing comprehensive analysis of backtesting results, detecting principle violations, and offering actionable improvement suggestions through a conversational interface.

## 🌟 Features

### 1. **Vector Database (ChromaDB)**
- Stores trading principles from legendary traders (Livermore, Minervini, Buffett, Dalio)
- Indexes news articles with timestamps and symbols
- Semantic search for relevant context retrieval

### 2. **Dual LLM Support with Failover**
- **Primary**: Google Gemini (gemini-pro)
- **Fallback**: Groq (llama3-70b-8192)
- Automatic failover on errors
- All free-tier APIs

### 3. **Comprehensive Backtest Analysis**
- **Performance Explanation**: Correlates returns with market events
- **Risk Assessment**: Analyzes Sharpe ratio, drawdowns, win rates
- **Market Context**: Links performance to retrieved news
- **Regime Analysis**: Performance breakdown by market conditions

### 4. **Trading Principle Violation Detection**
- Detects averaging down on losing positions
- Identifies missing/violated stop losses
- Flags overtrading (holding periods < 2 days)
- Checks position sizing violations
- LLM-powered analysis with specific trade examples

### 5. **Improvement Suggestion Engine**
- Regime-specific filters (ADX thresholds for ranging markets)
- Volatility-adjusted position sizing recommendations
- Options hedging strategies
- Risk management enhancements
- Parameter tuning guidance

### 6. **Conversational Interface**
- Context-aware Q&A about backtest results
- Query intent classification
- Trade filtering and analysis
- Strategy comparison
- Conversation history and persistence

## 📁 Directory Structure

```
RAG_Mentor/
├── __init__.py
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── example_usage.py             # Demo script
│
├── vector_db/                   # Vector database layer
│   ├── __init__.py
│   ├── embeddings.py           # HuggingFace sentence-transformers
│   └── chroma_manager.py       # ChromaDB operations
│
├── llm/                        # LLM integration
│   ├── __init__.py
│   ├── llm_client.py           # Gemini/Groq with failover
│   └── prompt_templates.py     # Specialized prompts
│
├── knowledge/                  # Knowledge base
│   ├── __init__.py
│   ├── trading_principles.json # 20 trading principles
│   └── knowledge_loader.py     # Ingestion pipeline
│
├── mentor/                     # Core mentor logic
│   ├── __init__.py
│   ├── rag_engine.py           # RAG retrieval + generation
│   ├── performance_analyzer.py # Backtest analysis
│   ├── principle_checker.py    # Violation detection
│   └── improvement_engine.py   # Suggestions generator
│
└── interface/                  # User interface
    ├── __init__.py
    ├── conversation_manager.py # Context & history management
    └── trading_mentor.py       # Main interface
```

## 🚀 Quick Start

### 1. Installation

```bash
cd RAG_Mentor
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file (or copy from `.env.example`):

```env
# LLM API Keys (at least one required)
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# ChromaDB Path
CHROMADB_PATH=./RAG_Mentor/chroma_db

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Configuration
TOP_K_RESULTS=5
```

**Get API Keys (Free Tier):**
- **Gemini**: https://makersuite.google.com/app/apikey
- **Groq**: https://console.groq.com/keys

### 3. Initialize Knowledge Base

```python
from RAG_Mentor.knowledge.knowledge_loader import KnowledgeLoader

# Load trading principles and sample news
loader = KnowledgeLoader()
results = loader.initialize_knowledge_base(include_sample_news=True)

print(f"Loaded {results['principles']} principles")
print(f"Loaded {results['news']} news articles")
```

### 4. Analyze a Backtest

```python
from RAG_Mentor.interface.trading_mentor import TradingMentor

# Initialize mentor
mentor = TradingMentor()

# Prepare backtest data
performance = {
    'total_return': 0.15,
    'sharpe_ratio': 1.2,
    'max_drawdown': 0.18,
    'win_rate': 0.58,
    'profit_factor': 1.5,
    'total_trades': 120
}

trades = [
    {
        'timestamp': datetime(2023, 6, 15, 10, 30),
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.0,
        'pnl': 250.0,
        'reason': 'Technical breakout'
    },
    # ... more trades
]

# Run comprehensive analysis
results = mentor.analyze_backtest(
    performance_summary=performance,
    trades=trades,
    symbols=['AAPL', 'MSFT', 'GOOGL']
)

# Access results
print(results['summary'])                    # Executive summary
print(results['performance_analysis'])       # Detailed analysis
print(results['violation_report'])           # Principle violations
print(results['improvement_suggestions'])    # Recommendations
```

### 5. Conversational Q&A

```python
# Ask follow-up questions
response = mentor.ask_question("Why did the strategy fail in March 2020?")
print(response)

response = mentor.ask_question("Show me all trades during volatile markets")
print(response)

response = mentor.ask_question("Compare this strategy to buy-and-hold")
print(response)
```

### 6. Run Demo

```bash
python RAG_Mentor/example_usage.py
```

## 📊 Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║              TRADING STRATEGY ANALYSIS SUMMARY                ║
╚═══════════════════════════════════════════════════════════════╝

Performance Overview:
- Total Return: 15.00%
- Sharpe Ratio: 1.20
- Maximum Drawdown: 18.00%
- Win Rate: 58.00%
- Total Trades: 120

Quick Assessment:
⚠️ Promising but Needs Work - Has potential, requires optimization

═══════════════════════════════════════════════════════════════

PRINCIPLE VIOLATIONS DETECTED:

1. Averaging Down (Severity: MODERATE)
   - Found 3 instances of adding to losing positions
   - Example: AAPL position increased at $135 after entry at $150
   - Violates: Jesse Livermore's principle

2. Stop Loss Issues (Severity: CRITICAL)
   - 8 trades with losses >10% (indicates no stop loss)
   - Example: TSLA loss of -15% on [date]
   - Violates: Mark Minervini's risk management

═══════════════════════════════════════════════════════════════

IMPROVEMENT SUGGESTIONS:

### 📊 Improve Sharpe Ratio
Suggestion: Implement volatility-adjusted position sizing
Rationale: Your Sharpe ratio of 1.20 could improve with dynamic sizing
Implementation: position_size = base_size * (avg_vol / current_vol)
Expected Impact: 15-25% improvement in risk-adjusted returns

### 🛡️ Reduce Drawdown
Suggestion: Add portfolio heat limits
Rationale: Max drawdown of 18% exceeds best practices
Implementation: Limit total portfolio risk to 6% at any time
Expected Impact: Reduce max drawdown to <12%
```

## 🔧 Configuration

### Embedding Models

The system uses **sentence-transformers** by default (free):
- `all-MiniLM-L6-v2` (default, 384 dimensions, fast)
- `all-mpnet-base-v2` (768 dimensions, higher quality)

Change in `.env`:
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### LLM Models

Configure in `.env`:
```env
GEMINI_MODEL=gemini-pro
GROQ_MODEL=llama3-70b-8192
```

## 🧠 Knowledge Base

### Trading Principles Database

20 curated principles from:
- **Jesse Livermore**: Cut losses short, never average down, trend following
- **Mark Minervini**: Stage analysis, 1-2% risk per trade, stop losses
- **Warren Buffett**: Concentration over diversification
- **Ray Dalio**: Systematic rules, volatility ≠ risk
- **Modern Portfolio Theory**: Volatility clustering, regime adaptation

### News Articles

Sample database includes major events:
- Fed rate hikes
- CPI/inflation reports
- Banking crises
- Earnings announcements
- VIX spikes

Add custom news:
```python
loader.add_news_from_dict([
    {
        "headline": "Market crashes on geopolitical tensions",
        "summary": "...",
        "timestamp": datetime(2024, 1, 15),
        "symbols": ["SPY", "QQQ"],
        "source": "Reuters"
    }
])
```

## 🔗 Integration with Backtesting Engine

```python
# Example: Integration with quant_strategy module
from quant_strategy.engine import BacktestEngine
from RAG_Mentor.interface.trading_mentor import TradingMentor

# Run backtest
engine = BacktestEngine(...)
backtest_results = engine.run(...)

# Extract data
performance = {
    'total_return': backtest_results.portfolio_state.total_return,
    'sharpe_ratio': backtest_results.metrics['sharpe_ratio'],
    'max_drawdown': backtest_results.portfolio_state.max_drawdown,
    # ... more metrics
}

trades = backtest_results.trade_log

# Analyze with RAG Mentor
mentor = TradingMentor()
analysis = mentor.analyze_backtest(performance, trades)
```

## 📝 API Reference

### TradingMentor

**Main Methods:**

- `analyze_backtest(performance, trades, symbols, regime_breakdown)` → Comprehensive analysis
- `ask_question(question)` → Conversational Q&A
- `reset_conversation()` → Clear conversation state
- `get_conversation_history()` → Retrieve chat history

### ChromaManager

**Vector DB Operations:**

- `add_trading_principle(principle, explanation, author, category)` → Add principle
- `add_news_article(headline, summary, timestamp, symbols)` → Add news
- `search_principles(query, top_k, category_filter)` → Search principles
- `search_news(query, top_k, start_date, end_date)` → Search news

## 🎯 Use Cases

1. **Strategy Development**: Understand why backtests fail and get specific improvements
2. **Risk Management**: Detect discipline violations and excessive risk-taking
3. **Performance Debugging**: Correlate drawdowns with market events
4. **Learning**: Access trading wisdom from legendary traders
5. **Decision Support**: Get regime-specific recommendations

## 🚨 Limitations

- LLM responses require API connectivity (free tier has rate limits)
- Embedding model needs ~500MB disk space (first run)
- ChromaDB stores data locally (~50MB for 100 documents)
- Sample news database is limited (add your own for production)

## 📚 Further Reading

- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Gemini API](https://ai.google.dev/docs)
- [Groq API](https://console.groq.com/docs)

## 📄 License

MIT License - See parent project for details.

---

**Built with ❤️ for the CF-AI-SDE quantitative trading system**
