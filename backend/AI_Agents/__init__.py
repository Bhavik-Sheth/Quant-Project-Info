"""
AI Agents Module

Multi-agent system for trading operations:
- Portfolio Management Agent
- Risk Monitoring Agent
- Market Analysis Agent
- Trade Execution Agent
- Communication Protocol
"""

from .base_agent import BaseAgent, AgentResponse
from .communication_protocol import AgentMessage, MessageRouter

try:
    from .agents import (
        PortfolioManagementAgent,
        RiskMonitoringAgent,
        MarketAnalysisAgent,
        TradeExecutionAgent,
    )
except ImportError:
    pass

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "AgentMessage",
    "MessageRouter",
    "PortfolioManagementAgent",
    "RiskMonitoringAgent",
    "MarketAnalysisAgent",
    "TradeExecutionAgent",
]
