"""
API Dependencies

Dependency injection for FastAPI endpoints.
"""

from logical_pipe import TradingSystemAPI
from fastapi import HTTPException

# Global trading API instance (initialized in main.py)
trading_api: TradingSystemAPI = None


def get_trading_api() -> TradingSystemAPI:
    """
    Get the trading API instance
    
    Returns:
        TradingSystemAPI instance
        
    Raises:
        HTTPException: If trading API is not initialized
    """
    if trading_api is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Trading API not initialized",
                "reason": "System startup failed - check server logs for details",
                "solution": "Check backend logs for configuration errors"
            }
        )
    return trading_api
