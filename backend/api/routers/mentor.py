"""
RAG Mentor Router

Endpoints for trading mentor guidance.
"""

from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_trading_api
from api.models.requests import MentorQueryRequest
from api.models.responses import MentorResponse, ErrorResponse
from logical_pipe import TradingSystemAPI
from datetime import datetime
import logging

router = APIRouter(prefix="/mentor", tags=["RAG Mentor"])
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=MentorResponse, responses={500: {"model": ErrorResponse}})
async def ask_mentor(request: MentorQueryRequest, api: TradingSystemAPI = Depends(get_trading_api)):
    """
    Ask RAG mentor for trading guidance
    
    Uses retrieval-augmented generation to answer questions about:
    - Trading strategies
    - Risk management
    - Performance analysis
    - Market conditions
    """
    try:
        logger.info(f"Mentor query: {request.question[:50]}...")
        
        answer = api.query_rag_mentor(
            question=request.question,
            symbol=request.symbol,
            context=request.context
        )
        
        return MentorResponse(
            status="success",
            answer=answer,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Mentor query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
