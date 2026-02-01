"""
Backtesting Router

Endpoints for strategy backtesting.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from api.models.requests import BacktestRequest
from api.models.responses import BacktestResponse, ErrorResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime
import logging

router = APIRouter(prefix="/backtest", tags=["Backtesting"])
logger = logging.getLogger(__name__)


@router.post("/run", response_model=BacktestResponse, responses={500: {"model": ErrorResponse}})
async def run_backtest(request: BacktestRequest, api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Run strategy backtest
    
    Executes complete backtesting with:
    - Realistic execution simulation
    - Risk management
    - Performance metrics
    - Trade history
    """
    try:
        logger.info(f"Running backtest for {request.symbol} with {request.strategy}")
        
        result = api.run_backtest(
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            config=request.config
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return BacktestResponse(
            status="success",
            symbol=result['symbol'],
            strategy=result['strategy'],
            performance=result['performance'],
            trades=result['trades'],
            date_range=result['date_range'],
            timestamp=result['timestamp']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
