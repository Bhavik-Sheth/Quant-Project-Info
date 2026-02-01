"""
Signal Generation Router

Endpoints for generating trading signals.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from api.models.requests import SignalRequest
from api.models.responses import SignalResponse, ErrorResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime
import logging

router = APIRouter(prefix="/signals", tags=["Signals"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=SignalResponse, responses={500: {"model": ErrorResponse}})
async def generate_signals(request: SignalRequest, api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Generate trading signals
    
    Uses specified strategy to generate signals:
    - rsi: RSI-based mean reversion
    - macd: MACD crossover
    - ml_enhanced: ML-powered predictions
    - multi_agent: Full AI agent orchestration
    """
    try:
        logger.info(f"Generating signals for {request.symbol} using {request.strategy}")
        
        result = api.generate_signals(
            symbol=request.symbol,
            strategy=request.strategy,
            lookback_period=request.lookback_period
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return SignalResponse(
            status="success",
            symbol=result['symbol'],
            strategy=result['strategy'],
            signals=result['signals'],
            timestamp=result['timestamp'],
            message=f"Generated {result['strategy']} signals"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies():
    """List available trading strategies"""
    return {
        "status": "success",
        "strategies": [
            {"name": "rsi", "description": "RSI-based mean reversion", "type": "technical"},
            {"name": "macd", "description": "MACD crossover", "type": "technical"},
            {"name": "ml_enhanced", "description": "ML-enhanced predictions", "type": "ml"},
            {"name": "multi_agent", "description": "AI agent orchestration", "type": "agent"}
        ]
    }
