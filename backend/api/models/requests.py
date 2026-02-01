"""
Request Models

Pydantic models for API requests.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class MarketDataRequest(BaseModel):
    """Request model for market data ingestion"""
    symbol: str = Field(..., description="Ticker symbol", example="AAPL")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)", example="2024-01-01")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)", example="2024-12-31")
    interval: str = Field(default="1d", description="Data timeframe", example="1d")


class SignalRequest(BaseModel):
    """Request model for signal generation"""
    symbol: str = Field(..., description="Ticker symbol", example="AAPL")
    strategy: str = Field(default="rsi", description="Strategy name (rsi, macd, ml_enhanced, multi_agent)", example="rsi")
    lookback_period: int = Field(default=100, description="Number of periods to analyze", example=100)


class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    symbol: str = Field(..., description="Ticker symbol", example="AAPL")
    strategy: str = Field(..., description="Strategy name", example="rsi")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)", example="2024-01-01")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)", example="2024-12-31")
    initial_capital: float = Field(default=100000.0, description="Starting capital", example=100000.0)
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional strategy configuration")


class AgentAnalysisRequest(BaseModel):
    """Request model for AI agent analysis"""
    symbol: str = Field(..., description="Ticker symbol", example="AAPL")
    agent_type: str = Field(
        default="full",
        description="Agent type: market_data, risk, sentiment, volatility, regime, full",
        example="full"
    )


class MentorQueryRequest(BaseModel):
    """Request model for RAG mentor queries"""
    question: str = Field(..., description="Trading question", example="How should I manage risk in volatile markets?")
    symbol: Optional[str] = Field(default=None, description="Optional symbol for context", example="AAPL")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional additional context")
