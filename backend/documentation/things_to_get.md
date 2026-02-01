# CF-AI-SDE: Complete Setup Checklist

This document provides a comprehensive checklist of everything needed to run the CF-AI-SDE trading system. Follow each section in order for a smooth setup experience.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [MongoDB Setup](#mongodb-setup)
3. [Python Environment](#python-environment)
4. [Dependencies Installation](#dependencies-installation)
5. [API Keys Acquisition](#api-keys-acquisition)
6. [ChromaDB Initialization](#chromadb-initialization)
7. [GPU Support (Optional)](#gpu-support-optional)
8. [Database Initialization](#database-initialization)
9. [Testing & Verification](#testing--verification)
10. [Troubleshooting](#troubleshooting)

---

## 1. System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (more for historical data)
- **Python**: 3.10 or 3.11 (3.12 not fully tested)

### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: SSD with 50GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (for faster ML training)
- **Network**: Stable internet for API calls

**Checklist:**
- [ ] Verify Python version: `python --version` (should show 3.10.x or 3.11.x)
- [ ] Check available RAM: `free -h` (Linux) or Task Manager (Windows)
- [ ] Ensure 5GB+ free disk space

---

## 2. MongoDB Setup

You need **one** of the following MongoDB options:

### Option A: MongoDB Atlas (Cloud - Recommended for Beginners)

1. **Create Account**
   - Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
   - Sign up for free (M0 cluster - 512MB free tier)

2. **Create Cluster**
   - Click "Build a Database" → "FREE" tier
   - Choose cloud provider & region (closest to you)
   - Cluster name: `cf-ai-sde-cluster` (or your choice)

3. **Configure Access**
   - Database Access → Add Database User
     - Username: `cf_admin` (example)
     - Password: Generate secure password
     - Database User Privileges: "Atlas admin"
   - Network Access → Add IP Address
     - Option 1: Add your current IP
     - Option 2: Allow access from anywhere: `0.0.0.0/0` (less secure, use for testing only)

4. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy connection string: `mongodb+srv://cf_admin:<password>@cf-ai-sde-cluster.xxxxx.mongodb.net/`
   - Replace `<password>` with your actual password
   - Save to `.env` as `MONGODB_URI`

**Checklist:**
- [ ] MongoDB Atlas account created
- [ ] Free M0 cluster deployed
- [ ] Database user created with password saved securely
- [ ] IP whitelist configured
- [ ] Connection string obtained and tested

### Option B: Local MongoDB (Advanced Users)

1. **Download MongoDB Community Server**
   - [MongoDB Download Center](https://www.mongodb.com/try/download/community)
   - Version 6.0+ recommended
   - Select your OS and download installer

2. **Install MongoDB**
   - **Windows**: Run `.msi` installer, enable "MongoDB Compass" (GUI) option
   - **macOS**: 
     ```bash
     brew tap mongodb/brew
     brew install mongodb-community@6.0
     ```
   - **Linux (Ubuntu/Debian)**:
     ```bash
     wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
     echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
     sudo apt-get update
     sudo apt-get install -y mongodb-org
     ```

3. **Start MongoDB Service**
   - **Windows**: MongoDB runs as service automatically after installation
   - **macOS**: `brew services start mongodb-community@6.0`
   - **Linux**: `sudo systemctl start mongod && sudo systemctl enable mongod`

4. **Verify Installation**
   ```bash
   mongo --version  # Should show MongoDB shell version 6.0.x
   mongosh          # Connect to local instance (exit with .exit)
   ```

5. **Connection String**
   - Local: `mongodb://localhost:27017`
   - Save to `.env` as `MONGODB_URI=mongodb://localhost:27017`

**Checklist:**
- [ ] MongoDB Community Server installed
- [ ] MongoDB service running (`mongod` process active)
- [ ] Can connect via `mongosh` command
- [ ] Connection string saved to `.env`

---

## 3. Python Environment

### Option A: Using venv (Recommended)

```bash
# Navigate to project root
cd CF-AI-SDE

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python  # macOS/Linux
where python  # Windows
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n cf-ai-sde python=3.11 -y

# Activate environment
conda activate cf-ai-sde

# Verify
python --version
```

**Checklist:**
- [ ] Virtual environment created
- [ ] Environment activated (see `(venv)` or `(cf-ai-sde)` in terminal prompt)
- [ ] Python version verified (3.10 or 3.11)

---

## 4. Dependencies Installation

### Install Core Dependencies

```bash
# Ensure virtual environment is activated first!

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# This will install:
# - yfinance (market data)
# - pandas, numpy, scipy (data processing)
# - ta, ta-lib (technical indicators - see note below)
# - scikit-learn, xgboost (ML models)
# - tensorflow/pytorch (deep learning)
# - pymongo (MongoDB driver)
# - chromadb (vector database)
# - google-generativeai, groq (LLM clients)
# - pyyaml, python-dotenv (configuration)
# - And 30+ other dependencies...
```

### Special Dependency: TA-Lib (Optional but Recommended)

TA-Lib provides advanced technical indicators. Installation varies by OS:

**Windows:**
1. Download wheel from [TA-Lib Unofficial](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Choose matching Python version: `TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl` (for Python 3.11, 64-bit)
2. Install: `pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl`

**macOS:**
```bash
# Install TA-Lib C library first
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

**Linux (Ubuntu/Debian):**
```bash
# Install dependencies
sudo apt-get install build-essential wget

# Download and install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Install Python wrapper
pip install TA-Lib
```

**If TA-Lib installation fails:**
- System will use `ta` library instead (subset of indicators)
- Feature engineering will still work with reduced indicator set

### Verify Installation

```bash
# Test critical imports
python -c "import yfinance; print('yfinance OK')"
python -c "import pymongo; print('pymongo OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import chromadb; print('chromadb OK')"
python -c "import google.generativeai; print('Gemini SDK OK')"

# Test TA-Lib (optional)
python -c "import talib; print('TA-Lib OK')" || echo "TA-Lib not available (using 'ta' library instead)"
```

**Checklist:**
- [ ] `requirements.txt` dependencies installed without errors
- [ ] Core imports verified (yfinance, pymongo, xgboost, chromadb, google.generativeai)
- [ ] TA-Lib installed (optional) or fallback to `ta` library

---

## 5. API Keys Acquisition

### Required API Keys

#### A. Google Gemini API (REQUIRED)

**Purpose**: Primary LLM for AI agents and RAG Mentor

**Steps:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Get API Key" → "Create API key"
4. Copy key (starts with `AIza...`)
5. Add to `.env`: `GEMINI_API_KEY=AIzaXXXXXXXXXXXXXXXXXXXX`

**Free Tier Limits:**
- 60 requests/minute
- 1,500 requests/day
- Sufficient for development and testing

**Checklist:**
- [ ] Google account created/logged in
- [ ] Gemini API key generated
- [ ] Key saved to `.env` file
- [ ] Verified key works: 
  ```bash
  python -c "from google import generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('Gemini OK')"
  ```

#### B. Groq API (OPTIONAL - Fallback LLM)

**Purpose**: Backup LLM when Gemini quota exceeded or unavailable

**Steps:**
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up (GitHub/Google login available)
3. Navigate to "API Keys" section
4. Click "Create API Key" → Name it "CF-AI-SDE"
5. Copy key (starts with `gsk_...`)
6. Add to `.env`: `GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXX`

**Free Tier Limits:**
- 30 requests/minute
- Fast inference speed (LPU architecture)

**Checklist:**
- [ ] Groq account created
- [ ] API key generated
- [ ] Key saved to `.env` file

### Optional API Keys

#### C. NewsAPI (Recommended for Sentiment Analysis)

**Purpose**: Sentiment agent - analyze news sentiment for tickers

**Steps:**
1. Go to [NewsAPI](https://newsapi.org/register)
2. Sign up with email
3. Confirm email and log in
4. Copy API key from dashboard
5. Add to `.env`: `NEWS_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXX`

**Free Tier Limits:**
- 100 requests/day
- News articles from past 30 days

**Note**: If not provided, SentimentAgent will be skipped gracefully.

**Checklist:**
- [ ] (Optional) NewsAPI account created
- [ ] (Optional) API key obtained
- [ ] (Optional) Key added to `.env`

#### D. HuggingFace Token (Optional)

**Purpose**: Access gated models like FinBERT for sentiment analysis

**Steps:**
1. Go to [HuggingFace](https://huggingface.co/join)
2. Sign up and verify email
3. Navigate to Settings → [Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token" → Name: "CF-AI-SDE" → Role: "Read"
5. Copy token (starts with `hf_...`)
6. Add to `.env`: `HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX`

**Checklist:**
- [ ] (Optional) HuggingFace account created
- [ ] (Optional) Access token generated
- [ ] (Optional) Token added to `.env`

#### E. Economic Calendar API (Optional)

**Purpose**: Macro agent - economic events and data releases

**Options:**
- [TradingEconomics](https://tradingeconomics.com/api) (paid)
- [Forex Factory](https://www.forexfactory.com/) (no official API - web scraping)
- [Alpha Vantage](https://www.alphavantage.co/support/#api-key) (free tier: 5 calls/min)

**Note**: If not provided, MacroAgent will be skipped.

**Checklist:**
- [ ] (Optional) Economic API selected
- [ ] (Optional) API key obtained
- [ ] (Optional) Key added to `.env`: `ECONOMIC_CALENDAR_API_KEY=XXXXX`

### Environment File Setup

Create `.env` file in project root:

```bash
# Copy template
cp .env.example .env

# Edit .env file and replace placeholders with actual keys
# Use any text editor: nano .env, vim .env, or VS Code
```

**Final .env should look like:**
```env
# Required
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXX
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=cf_ai_sde

# Optional but recommended
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXX
NEWS_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX

# Optional
ECONOMIC_CALENDAR_API_KEY=
VAR_LIMIT=0.05
DRAWDOWN_LIMIT=0.10
CHROMADB_PATH=./RAG_Mentor/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=5
LOG_LEVEL=INFO
```

**Checklist:**
- [ ] `.env` file created from `.env.example`
- [ ] At least `GEMINI_API_KEY` added (required)
- [ ] MongoDB connection details added
- [ ] Optional keys added if available
- [ ] `.env` file **NOT** committed to git (verify `.gitignore` includes `.env`)

---

## 6. ChromaDB Initialization

ChromaDB is used by RAG Mentor for vector-based knowledge retrieval.

### Automatic Initialization

ChromaDB will auto-initialize on first run, but you can pre-populate it:

```bash
# Navigate to RAG_Mentor directory
cd RAG_Mentor

# Run knowledge loader
python -c "from knowledge.knowledge_loader import KnowledgeLoader; loader = KnowledgeLoader(); loader.load_principles(); print('ChromaDB initialized')"

# Or use the example script
python example_usage.py
```

### Manual Verification

```bash
# Check if ChromaDB directory created
ls -la RAG_Mentor/chroma_db  # macOS/Linux
dir RAG_Mentor\chroma_db      # Windows

# Should see SQLite database files
```

### Load Trading Principles

```bash
# Ensure trading_principles.json exists
ls RAG_Mentor/knowledge/trading_principles.json

# If missing, it will be created with defaults on first run
```

**Checklist:**
- [ ] ChromaDB directory exists: `RAG_Mentor/chroma_db/`
- [ ] Trading principles loaded (verify no errors when running example_usage.py)
- [ ] Can query ChromaDB: 
  ```bash
  python -c "from RAG_Mentor.interface.trading_mentor import TradingMentor; mentor = TradingMentor(); print('ChromaDB accessible')"
  ```

---

## 7. GPU Support (Optional)

GPU acceleration speeds up ML model training (5-10x faster).

### Check GPU Availability

```bash
# Check if NVIDIA GPU detected
nvidia-smi  # Should show GPU info if available

# Check CUDA version
nvcc --version
```

### Install CUDA Toolkit (NVIDIA GPUs only)

**Windows/Linux:**
1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install with default settings
3. Verify: `nvcc --version`

**macOS:**
- macOS doesn't support CUDA (use CPU or consider M1/M2 with MPS backend)

### Install TensorFlow GPU / PyTorch GPU

**TensorFlow:**
```bash
pip uninstall tensorflow  # Remove CPU version
pip install tensorflow-gpu==2.13.0
```

**PyTorch (check [PyTorch website](https://pytorch.org/get-started/locally/) for exact command):**
```bash
# Example for CUDA 11.8:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Usage

```bash
# TensorFlow
python -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"

# PyTorch
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Checklist:**
- [ ] (Optional) NVIDIA GPU detected
- [ ] (Optional) CUDA Toolkit installed
- [ ] (Optional) TensorFlow/PyTorch GPU versions installed
- [ ] (Optional) GPU accessibility verified

**Note**: System works fine on CPU, just slower for training.

---

## 8. Database Initialization

Run automated database setup script:

```bash
# From project root directory
python setup_database.py
```

**What this script does:**
1. Tests MongoDB connection
2. Creates required collections:
   - `market_data_raw`
   - `market_data_clean`
   - `market_features`
   - `agent_outputs`
   - `agent_memory`
   - `backtest_results`
   - `strategy_performance`
   - `ml_models`
   - `rag_knowledge`
3. Creates indexes for query optimization:
   - `market_data_raw`: `(ticker, timestamp)`
   - `market_data_clean`: `(ticker, timestamp)`
   - `market_features`: `(ticker, timestamp, feature_name)`
   - `agent_outputs`: `(ticker, timestamp, agent_name)`
   - `agent_memory`: `(agent_name, timestamp)`, `performance_weight`
   - `backtest_results`: `(strategy_name, timestamp)`
   - `strategy_performance`: `(strategy_name, metric_name)`
   - `ml_models`: `(model_type, version)`, `trained_at`, `status`
   - `rag_knowledge`: `(document_id)`
4. Tests write/read/delete operations
5. Verifies index creation

**Expected Output:**
```
=== CF-AI-SDE Database Setup ===

Step 1/5: Loading environment and testing connection...
✓ Environment variables loaded
✓ MongoDB connection successful

Step 2/5: Creating indexes...
✓ Indexes created successfully

Step 3/5: Verifying collections...
✓ Collections ready (may be created on first insert)

Step 4/5: Testing database operations...
✓ Write operation successful
✓ Read operation successful
✓ Delete operation successful

Step 5/5: Verifying index counts...
✓ market_data_raw: 1 indexes
✓ market_data_clean: 1 indexes
✓ market_features: 1 indexes
[... more collections ...]

=== Setup Complete! ===
Database is ready for use.
```

**If setup fails:**
- Check MongoDB service is running: `mongosh` (should connect)
- Verify `.env` file has correct `MONGODB_URI`
- Check MongoDB logs: 
  - Windows: `C:\Program Files\MongoDB\Server\6.0\log\mongod.log`
  - Linux: `/var/log/mongodb/mongod.log`
  - macOS: `/usr/local/var/log/mongodb/mongo.log`

**Checklist:**
- [ ] `setup_database.py` runs without errors
- [ ] All collections created (verify in MongoDB Compass or mongosh)
- [ ] Indexes created successfully
- [ ] Test operations pass (write/read/delete)

---

## 9. Testing & Verification

### Quick Smoke Test

```bash
# Test imports and basic functionality
python -c "from logical_pipe import TradingSystemAPI; api = TradingSystemAPI('./config.yaml'); print('Health Check:', api.health_check())"
```

**Expected output:**
```python
{
    'status': 'healthy',
    'mongodb_connected': True,
    'config_loaded': True,
    'modules_loaded': {
        'data_pipeline': True,
        'model_manager': True,
        'agent_orchestrator': True,
        'strategy_engine': True,
        'analysis_interface': True
    },
    'timestamp': '2024-01-15T10:30:45.123456'
}
```

### Run Example Pipeline

```bash
# Test data ingestion and feature engineering
python -c "
from logical_pipe import TradingSystemAPI
api = TradingSystemAPI('./config.yaml')

# Ingest data for AAPL
result = api.run_partial_pipeline(
    stages=['data_ingestion', 'feature_engineering'],
    ticker='AAPL',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
print('Pipeline result:', result)
"
```

### Test Individual Modules

```bash
# Test Data Pipeline
cd Data-inges-fe
python quick_run.py

# Test ML Models
cd ../ML_Models
python -c "from direction_pred import train_direction_model; print('ML models importable')"

# Test AI Agents
cd ../AI_Agents
python example_usage.py

# Test Strategy Engine
cd ../quant_strategy
python example_usage.py

# Test RAG Mentor
cd ../RAG_Mentor
python example_usage.py
```

### Run Full System Test

```bash
# From project root
python -c "
from logical_pipe import TradingSystemAPI
import pandas as pd

api = TradingSystemAPI('./config.yaml')

# Run full pipeline for a short period
result = api.run_full_pipeline(
    ticker='SPY',
    start_date='2024-01-01',
    end_date='2024-01-10',
    strategy_name='RSI_Strategy',
    analyze_results=True
)

print('Full pipeline completed!')
print('Backtest results:', result.get('backtest_results', {}).get('metrics', {}))
"
```

**Checklist:**
- [ ] Health check passes with all modules loaded
- [ ] Partial pipeline (data ingestion + features) works
- [ ] Individual module tests pass
- [ ] Full pipeline completes without errors
- [ ] MongoDB collections populated with data

---

## 10. Troubleshooting

### Common Issues & Solutions

#### Issue: `ModuleNotFoundError: No module named 'xxx'`

**Cause**: Dependency not installed or wrong Python environment

**Solution**:
```bash
# Verify virtual environment active
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt

# If specific module missing:
pip install <module_name>
```

---

#### Issue: `pymongo.errors.ServerSelectionTimeoutError`

**Cause**: MongoDB not running or connection string incorrect

**Solution**:
```bash
# Check MongoDB service status
# Windows: Check Services app for "MongoDB Server"
# macOS: brew services list | grep mongodb
# Linux: sudo systemctl status mongod

# Start MongoDB if stopped
# Windows: Services → MongoDB Server → Start
# macOS: brew services start mongodb-community@6.0
# Linux: sudo systemctl start mongod

# Verify connection string in .env
# Should be: mongodb://localhost:27017 (local) or mongodb+srv://... (Atlas)

# Test connection
mongosh  # Should connect without errors
```

---

#### Issue: `google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded`

**Cause**: Gemini API rate limit reached (60 req/min or 1,500/day)

**Solution**:
1. **Wait**: Rate limits reset after 1 minute (for per-minute) or 24 hours (for daily)
2. **Enable Groq fallback**: Ensure `GROQ_API_KEY` set in `.env`
3. **Reduce frequency**: Add delays between API calls in agent orchestration
4. **Upgrade**: Consider [Gemini API paid tier](https://ai.google.dev/pricing) for higher limits

---

#### Issue: `ImportError: DLL load failed` or `OSError: [WinError 126]` (Windows, TA-Lib)

**Cause**: TA-Lib C library not installed properly

**Solution**:
1. Download correct wheel from [TA-Lib Unofficial](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
2. Match Python version (cp311 = Python 3.11) and architecture (amd64 = 64-bit)
3. Install: `pip install <downloaded_wheel_file>.whl`
4. **If still fails**: System will use `ta` library instead (reduced indicator set)

---

#### Issue: `chromadb.errors.ChromaError: Could not connect to database`

**Cause**: ChromaDB path doesn't exist or permissions issue

**Solution**:
```bash
# Create ChromaDB directory
mkdir -p RAG_Mentor/chroma_db  # macOS/Linux
md RAG_Mentor\chroma_db         # Windows

# Set correct permissions
chmod 755 RAG_Mentor/chroma_db  # macOS/Linux

# Reinitialize ChromaDB
python -c "from RAG_Mentor.knowledge.knowledge_loader import KnowledgeLoader; KnowledgeLoader().load_principles()"
```

---

#### Issue: `tensorflow` or `torch` not using GPU

**Cause**: CUDA not installed or GPU drivers outdated

**Solution**:
1. Update NVIDIA drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Install CUDA Toolkit 11.8: [CUDA Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)
3. Reinstall GPU version of TensorFlow/PyTorch
4. Verify:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

**Note**: System works on CPU - GPU only speeds up training.

---

#### Issue: `KeyError: 'GEMINI_API_KEY'` or environment variables not loading

**Cause**: `.env` file not in project root or missing variables

**Solution**:
```bash
# Verify .env file location
ls -la .env  # Should be in CF-AI-SDE/ root directory

# Check file contents
cat .env  # macOS/Linux
type .env  # Windows

# Ensure required keys present:
# - GEMINI_API_KEY
# - MONGODB_URI
# - MONGODB_DATABASE

# Reload environment (if running Python interactively)
from dotenv import load_dotenv
load_dotenv(override=True)
```

---

#### Issue: `ValueError: Could not find config file` when running `TradingSystemAPI`

**Cause**: `config.yaml` not in expected location

**Solution**:
```bash
# Verify config.yaml exists
ls -la config.yaml  # Should be in CF-AI-SDE/ root

# Or provide absolute path
python -c "from logical_pipe import TradingSystemAPI; api = TradingSystemAPI('/full/path/to/config.yaml')"
```

---

#### Issue: Data ingestion fails with `yfinance` errors

**Cause**: Yahoo Finance API rate limiting or network issues

**Solution**:
1. **Rate limiting**: Add delays between ticker downloads
2. **Network issues**: Check internet connection, try again later
3. **Ticker not found**: Verify ticker symbol is correct (e.g., `AAPL`, not `Apple`)
4. **Alternatives**: Consider paid data providers (Alpha Vantage, Polygon.io) for production

---

#### Issue: MongoDB collections not created after running `setup_database.py`

**Cause**: Collections are created lazily on first insert (MongoDB design)

**Solution**:
- This is normal behavior! Collections appear after first data insertion.
- Verify collections exist after running data ingestion:
  ```bash
  mongosh
  use cf_ai_sde
  show collections  # Should show collections after data ingestion
  ```

---

### Getting Help

If issues persist:

1. **Check logs**: Look for errors in terminal output
2. **Enable debug logging**: Set `LOG_LEVEL=DEBUG` in `.env`
3. **Search issues**: Check GitHub Issues for similar problems
4. **Ask for help**: Open new issue with:
   - Error message (full traceback)
   - Python version: `python --version`
   - OS version
   - Steps to reproduce
   - `.env` contents (REDACTED - remove actual API keys!)

---

## Summary Checklist

Print this and check off as you go!

### Prerequisites
- [ ] Python 3.10 or 3.11 installed
- [ ] 5GB+ free disk space
- [ ] 8GB+ RAM available

### MongoDB
- [ ] MongoDB installed (local) OR Atlas cluster created (cloud)
- [ ] MongoDB service running
- [ ] Connection string added to `.env`

### Python Environment
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from `requirements.txt`
- [ ] TA-Lib installed (optional)

### API Keys
- [ ] Gemini API key obtained and added to `.env` (REQUIRED)
- [ ] Groq API key obtained and added to `.env` (recommended)
- [ ] NewsAPI key obtained (optional)
- [ ] HuggingFace token obtained (optional)

### Configuration
- [ ] `.env` file created from `.env.example`
- [ ] All required environment variables set
- [ ] `.env` not committed to version control

### Initialization
- [ ] ChromaDB initialized (directory exists)
- [ ] Database setup script run successfully: `python setup_database.py`
- [ ] Indexes created in MongoDB

### Verification
- [ ] Health check passes: `TradingSystemAPI('./config.yaml').health_check()`
- [ ] Can import all modules without errors
- [ ] Partial pipeline test passes
- [ ] MongoDB collections populated after test

### Optional Enhancements
- [ ] GPU support configured (NVIDIA only)
- [ ] TA-Lib installed for advanced indicators
- [ ] Economic calendar API configured

---

## Next Steps

Once all checklist items complete:

1. **Read the README**: Familiarize yourself with system architecture
2. **Explore examples**: 
   - `quant_strategy/example_usage.py` - Strategy backtesting
   - `AI_Agents/example_usage.py` - Multi-agent system
   - `RAG_Mentor/example_usage.py` - Trading mentor queries
3. **Run a full backtest**: Use `TradingSystemAPI.run_full_pipeline()`
4. **Customize strategies**: Modify `quant_strategy/strategies/` or create new ones
5. **Train ML models**: Use `ModelManager` to train on your own data
6. **Deploy to production**: See deployment guides (Docker, cloud hosting)

**Congratulations!** Your CF-AI-SDE system is ready to use. 🚀

---

**Last Updated**: 2024-01-15  
**Maintainer**: CF-AI-SDE Team  
**Support**: Open an issue on GitHub for help
