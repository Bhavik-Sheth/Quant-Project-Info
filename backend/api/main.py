"""
CF-AI-SDE Trading System API - Main Application

FastAPI app with all routers integrated.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# Import dependencies
import api.dependencies as deps
from logical_pipe import TradingSystemAPI, ConfigLoader

# Import routers
from api.routers import health, data, signals, backtest, agents, mentor, config, ml_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 CF-AI-SDE Trading System API Starting...")
    logger.info("=" * 70)
    
    try:
        # TradingSystemAPI creates its own ConfigLoader internally
        deps.trading_api = TradingSystemAPI("config.yaml")
        logger.info("✅ Trading System API initialized successfully")
        
        # Check health
        health = deps.trading_api.health_check()
        for component, status in health.items():
            icon = "✅" if status == "ok" else "⚠️"
            logger.info(f"{icon} {component}: {status}")
    
    except FileNotFoundError as e:
        logger.error("=" * 70)
        logger.error("❌ CRITICAL: Configuration file not found!")
        logger.error(str(e))
        logger.error("=" * 70)
        logger.error("💡 Solution:")
        logger.error("   1. Ensure config.yaml exists in backend/ directory")
        logger.error("   2. Check the paths shown above")
        logger.error("=" * 70)
        # Don't raise - let server start but all endpoints will return 503
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ CRITICAL: Trading API initialization failed!")
        logger.error(f"Error: {e}")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error("=" * 70)
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        logger.warning("⚠️  API will start but endpoints will return 503 errors")
    
    logger.info("=" * 70)
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("🔧 Health Check: http://localhost:8000/health")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("Shutting down CF-AI-SDE Trading System API...")


# Initialize FastAPI app
app = FastAPI(
    title="CF-AI-SDE Trading System",
    description="""
    ## AI-Powered Multi-Agent Trading System
    
    Complete trading system with:
    - **Data Ingestion**: Yahoo Finance integration with 70+ technical indicators
    - **ML Models**: Direction prediction, volatility forecasting, regime detection
    - **AI Agents**: 7 specialized agents for market analysis
    - **Backtesting**: Realistic execution simulation with risk management
    - **RAG Mentor**: Trading knowledge Q&A system
    
    ### Quick Start
    
    1. Check system health: `GET /health`
    2. Ingest data: `POST /data/ingest`
    3. Generate signals: `POST /signals/generate`
    4. Run backtest: `POST /backtest/run`
    5. Ask mentor: `POST /mentor/ask`
    
    ### Storage
    
    - Primary: MongoDB for production use
    - Fallback: TinyDB (automatic) stored in `data/fallback/`
    
    ### Documentation
    
    - Interactive API docs: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI spec: `/openapi.json`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(data.router)
app.include_router(signals.router)
app.include_router(backtest.router)
app.include_router(agents.router)
app.include_router(mentor.router)
app.include_router(config.router)
app.include_router(ml_models.router)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
