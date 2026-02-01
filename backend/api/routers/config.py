"""
Configuration Router

Endpoints for system configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from logical_pipe import TradingSystemAPI
import logging

router = APIRouter(prefix="/config", tags=["Configuration"])
logger = logging.getLogger(__name__)


@router.get("/")
async def get_config(api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Get system configuration
    
    Returns sanitized configuration (sensitive data removed).
    """
    try:
        config = api.get_config()
        
        return {
            "status": "success",
            "config": config,
            "message": "Configuration retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
