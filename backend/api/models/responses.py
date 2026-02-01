"""
Response Models

Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall status (healthy/degraded/error)", example="healthy")
    timestamp: str = Field(..., description="Check timestamp")
    components: Dict[str, str] = Field(..., description="Component health statuses")
    message: str = Field(..., description="Status message")


class MarketDataResponse(BaseModel):
    """Market data response"""
    status: str = Field(..., example="success")
    symbol: str = Field(..., description="Ticker symbol")
    records: int = Field(..., description="Number of records")
    date_range: Optional[str] = Field(default=None, description="Date range")
    storage: Optional[str] = Field(default=None, description="Storage location (mongodb/tinydb)")
    message: Optional[str] = Field(default=None, description="Status message")


class SignalResponse(BaseModel):
    """Signal generation response"""
    status: str = Field(..., example="success")
    symbol: str = Field(..., description="Ticker symbol")
    strategy: str = Field(..., description="Strategy used")
    signals: Dict[str, Any] = Field(..., description="Generated signals")
    timestamp: str = Field(..., description="Signal timestamp")
    message: Optional[str] = Field(default=None, description="Status message")


class BacktestResponse(BaseModel):
    """Backtest results response"""
    status: str = Field(..., example="success")
    symbol: str = Field(..., description="Ticker symbol")
    strategy: str = Field(..., description="Strategy used")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    trades: List[Dict[str, Any]] = Field(..., description="Trade history")
    date_range: str = Field(..., description="Backtest date range")
    timestamp: str = Field(..., description="Completion timestamp")


class AgentAnalysisResponse(BaseModel):
    """Agent analysis response"""
    status: str = Field(..., example="success")
    symbol: str = Field(..., description="Ticker symbol")
    agent_type: str = Field(..., description="Agent type used")
    analysis: Any = Field(..., description="Analysis results")
    timestamp: str = Field(..., description="Analysis timestamp")
    message: Optional[str] = Field(default=None, description="Status message")


class MentorResponse(BaseModel):
    """RAG mentor response"""
    status: str = Field(..., example="success")
    answer: str = Field(..., description="Mentor's answer")
    timestamp: str = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = Field(default="error", example="error")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
