// API Type Definitions
// These match the backend Pydantic models

export interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SignalRequest {
  symbol: string;
  timeframe: string;
  strategy: string;
  start_date?: string;
  end_date?: string;
  strategy_params?: Record<string, any>;
}

export interface Signal {
  timestamp: string;
  signal: 'buy' | 'sell' | 'hold';
  strength: number;
  price: number;
  metadata?: Record<string, any>;
}

export interface BacktestRequest {
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  strategy: string;
  initial_capital: number;
  strategy_params?: Record<string, any>;
}

export interface BacktestResult {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  equity_curve: Array<{timestamp: string; value: number}>;
}

export interface AgentRequest {
  symbol: string;
  timeframe: string;
  agent_type: 'market_data' | 'risk' | 'sentiment' | 'volatility' | 'portfolio' | 'full';
  context?: Record<string, any>;
}

export interface AgentResponse {
  agent_type: string;
  analysis: Record<string, any>;
  recommendations: string[];
  confidence: number;
  timestamp: string;
}

export interface MentorRequest {
  question: string;
  context?: Record<string, any>;
}

export interface MentorResponse {
  answer: string;
  sources: string[];
  confidence: number;
}

export interface DataIngestRequest {
  symbol: string;
  timeframe: string;
  start_date?: string;
  end_date?: string;
  source?: string;
}

export interface MLPredictionRequest {
  symbol: string;
  timeframe: string;
  features?: Record<string, any>;
}

export interface DirectionPrediction {
  direction: 'up' | 'down' | 'neutral';
  probability: number;
  confidence: number;
  features_importance?: Record<string, number>;
}

export interface VolatilityForecast {
  forecast: number[];
  timestamps: string[];
  model: string;
  confidence_intervals?: Array<{lower: number; upper: number}>;
}

export interface RegimeClassification {
  regime: string;
  probability: number;
  regime_probabilities: Record<string, number>;
}

export interface ApiError {
  detail: string;
  status: number;
  timestamp: string;
}
