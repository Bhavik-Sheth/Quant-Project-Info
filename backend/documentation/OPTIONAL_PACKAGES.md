# Optional Packages for Full Functionality

## Currently Missing (Non-Critical)

These packages have graceful fallbacks in the codebase. The system works without them.

### 1. arch - For GARCH Volatility Models
**Used in**: `ML_Models/Volatility_Forecasting.py`

```bash
pip install arch
```

**Impact if missing**: 
- VolatilityAgent uses simple fallback predictions
- GARCH models not available
- System continues working with alternative volatility estimates

**When to install**: 
- If you need advanced volatility forecasting
- For GARCH-based market analysis
- Before using VolatilityAgent in production

---

### 2. codecarbon - For Carbon Emissions Tracking
**Used in**: `RAG_Mentor/interface/trading_mentor.py`

```bash
pip install codecarbon
```

**Impact if missing**:
- Carbon footprint tracking disabled
- RAG Mentor works normally without it
- No environmental impact metrics

**When to install**:
- If you want to track AI model carbon emissions
- For ESG reporting requirements
- Optional for development

---

### 3. LangChain & LangGraph - For Advanced Orchestration
**Used in**: `quant_strategy/engine.py`, `AI_Agents/agents.py`

```bash
pip install langchain langgraph langchain-community
```

**Impact if missing**:
- Uses simple orchestration fallback
- Basic agent coordination still works
- No advanced LLM-based routing

**When to install**:
- For complex multi-agent workflows
- If using LLM-based strategy generation
- Production deployments with advanced features

---

### 4. Transformers Compatibility Issue
**Issue**: Python 3.13 has compatibility issues with some transformers features

**Affected modules**:
- `RAG_Mentor/mentor/rag_engine.py`
- `RAG_Mentor/vector_db/chroma_manager.py`

**Solutions**:
1. **Option A**: Use Python 3.10 or 3.11 (recommended)
   ```bash
   # Create new conda environment
   conda create -n cf-ai-sde python=3.11
   conda activate cf-ai-sde
   pip install -r requirements.txt
   ```

2. **Option B**: Wait for library updates
   - transformers will likely fix Python 3.13 compatibility soon
   - Check: https://github.com/huggingface/transformers/issues

3. **Option C**: Use without RAG embeddings (current fallback)
   - RAG LLM client still works
   - Vector embeddings use fallback

**When to fix**:
- Before using RAG Mentor embeddings feature
- If you need semantic search in trading knowledge base
- Not critical for initial FastAPI development

---

## Installation Priority

### High Priority (Install Soon)
```bash
# For full ML functionality
pip install arch
```

### Medium Priority (Optional Features)
```bash
# For advanced orchestration
pip install langchain langgraph langchain-community

# For carbon tracking
pip install codecarbon
```

### Low Priority (Development Convenience)
- Python 3.11 downgrade for transformers (only if using RAG embeddings)

---

## Verify Installation

After installing packages, run:
```bash
cd backend
python test_imports.py
```

Should show: "🎉 All imports successful! Backend is ready for FastAPI."

---

## Current System Status

✅ **All critical functionality works without these packages**
✅ **Graceful fallbacks implemented**
✅ **Ready for FastAPI development**

Install these packages when you need specific features, not before.
