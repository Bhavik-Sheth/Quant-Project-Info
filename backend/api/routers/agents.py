"""
AI Agents Router

Endpoints for AI agent analysis.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from api.models.requests import AgentAnalysisRequest
from api.models.responses import AgentAnalysisResponse, ErrorResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime
import logging

router = APIRouter(prefix="/agents", tags=["AI Agents"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AgentAnalysisResponse, responses={500: {"model": ErrorResponse}})
async def analyze_with_agents(request: AgentAnalysisRequest, api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Run AI agent analysis
    
    Analyzes market conditions using:
    - market_data: Market data anomaly detection
    - risk: Risk monitoring
    - sentiment: News sentiment analysis
    - volatility: Volatility forecasting
    - regime: Market regime detection
    - full: Complete multi-agent orchestration
    """
    try:
        logger.info(f"Running agent analysis for {request.symbol}")
        
        result = api.run_agent_analysis(
            symbol=request.symbol,
            agent_type=request.agent_type
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return AgentAnalysisResponse(
            status="success",
            symbol=result['symbol'],
            agent_type=result['agent_type'],
            analysis=result.get('analysis') or result.get('individual_analyses'),
            timestamp=result['timestamp'],
            message=f"Completed {result['agent_type']} analysis"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_agents():
    """List available AI agents"""
    return {
        "status": "success",
        "agents": [
            {"name": "market_data", "description": "Market anomaly detection"},
            {"name": "risk", "description": "Risk monitoring"},
            {"name": "sentiment", "description": "News sentiment analysis"},
            {"name": "volatility", "description": "Volatility forecasting"},
            {"name": "regime", "description": "Market regime detection"},
            {"name": "full", "description": "Complete multi-agent orchestration"}
        ]
    }
