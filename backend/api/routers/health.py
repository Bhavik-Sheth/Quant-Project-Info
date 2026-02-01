"""
Health Check Router

Endpoints for system health monitoring.
"""

from fastapi import APIRouter, Depends
from api.dependencies import get_trading_api
from api.models.responses import HealthResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime

router = APIRouter(prefix="", tags=["Health"])


@router.get("/", tags=["Root"])
async def root():
    """API welcome message"""
    return {
        "message": "CF-AI-SDE Trading System API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(api: TradingSystemAPI = Depends(get_trading_api)):
    """
    System health check
    
    Returns status of all components:
    - Config loaded
    - Database connectivity (MongoDB/TinyDB)
    - LLM APIs status
    - Data pipeline status
    - AI agents status
    - Strategy engine status
    - RAG mentor status
    """
    health_status = api.health_check()
    
    # Determine overall status
    statuses = list(health_status.values())
    if all(s == 'ok' for s in statuses):
        overall_status = "healthy"
    elif any('error' in s for s in statuses):
        overall_status = "error"
    else:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=health_status,
        message=f"System is {overall_status}"
    )
