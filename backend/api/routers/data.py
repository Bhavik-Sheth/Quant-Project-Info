"""
Data Ingestion Router

Endpoints for market data ingestion and retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from api.models.requests import MarketDataRequest
from api.models.responses import MarketDataResponse, ErrorResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime
import logging

router = APIRouter(prefix="/data", tags=["Data"])
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=MarketDataResponse, responses={500: {"model": ErrorResponse}})
async def ingest_data(request: MarketDataRequest, api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Ingest market data for a symbol
    
    Fetches OHLCV data from Yahoo Finance, engineers features,
    validates quality, and stores in MongoDB or TinyDB fallback.
    """
    try:
        logger.info(f"Ingesting data for {request.symbol}")
        
        result = api.ingest_market_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.interval
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return MarketDataResponse(
            status="success",
            symbol=result['symbol'],
            records=result['records'],
            date_range=result.get('date_range'),
            storage=result.get('storage'),
            message=f"Successfully ingested {result['records']} records"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{symbol}")
async def get_latest_data(
    symbol: str,
    limit: int = 100,
    api: TradingSystemAPI = Depends(get_trading_api)
):
    """
    Get latest market data for a symbol
    
    Retrieves recent data from MongoDB or TinyDB fallback.
    """
    try:
        data = api.get_market_data(symbol=symbol, limit=limit)
        
        return {
            "status": "success",
            "symbol": symbol,
            "count": len(data),
            "data": data
        }
    
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
