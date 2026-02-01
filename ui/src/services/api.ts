import axios, { AxiosInstance, AxiosError } from 'axios';
import { toast } from 'sonner';
import type {
  MarketData,
  SignalRequest,
  Signal,
  BacktestRequest,
  BacktestResult,
  AgentRequest,
  AgentResponse,
  MentorRequest,
  MentorResponse,
  DataIngestRequest,
  MLPredictionRequest,
  DirectionPrediction,
  VolatilityForecast,
  RegimeClassification,
  ApiError,
} from '@/types/api';

// Base API URL from environment or fallback
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - add auth tokens if needed
apiClient.interceptors.request.use(
  (config) => {
    // Add API key if available
    const apiKey = process.env.NEXT_PUBLIC_API_KEY;
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle errors globally
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    // Handle different error scenarios
    if (error.response) {
      // Server responded with error
      const message = error.response.data?.detail || 'An error occurred';
      console.error('API Error:', message, error.response.status);
      
      // Don't show toast for certain status codes that will be handled locally
      if (error.response.status !== 404) {
        toast.error(`API Error: ${message}`);
      }
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message);
      toast.error('Network error - cannot connect to backend');
    } else {
      // Something else happened
      console.error('Error:', error.message);
      toast.error('An unexpected error occurred');
    }
    
    return Promise.reject(error);
  }
);

// API Service Class
class ApiService {
  // Health Check
  async health(): Promise<{ status: string; timestamp: string }> {
    const response = await apiClient.get('/health');
    return response.data;
  }

  // Data Management
  async ingestData(request: DataIngestRequest): Promise<{ message: string; records_inserted: number }> {
    const response = await apiClient.post('/data/ingest', request);
    toast.success(`Data ingested: ${response.data.records_inserted} records`);
    return response.data;
  }

  async getLatestData(symbol: string, timeframe: string = '1d', limit: number = 100): Promise<MarketData[]> {
    const response = await apiClient.get('/data/latest', {
      params: { symbol, timeframe, limit },
    });
    return response.data;
  }

  // Signal Generation
  async generateSignals(request: SignalRequest): Promise<{ signals: Signal[]; strategy: string }> {
    const response = await apiClient.post('/signals/generate', request);
    toast.success(`Generated ${response.data.signals.length} signals`);
    return response.data;
  }

  async listStrategies(): Promise<{ strategies: Array<{ name: string; description: string }> }> {
    const response = await apiClient.get('/signals/strategies');
    return response.data;
  }

  // Backtesting
  async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
    toast.info('Running backtest...');
    const response = await apiClient.post('/backtest/run', request);
    toast.success('Backtest completed successfully');
    return response.data;
  }

  // AI Agents
  async analyzeWithAgent(request: AgentRequest): Promise<AgentResponse | AgentResponse[]> {
    const response = await apiClient.post('/agents/analyze', request);
    toast.success(`Agent analysis complete`);
    return response.data;
  }

  async listAgents(): Promise<{ agents: Array<{ type: string; description: string }> }> {
    const response = await apiClient.get('/agents/list');
    return response.data;
  }

  // RAG Mentor
  async askMentor(request: MentorRequest): Promise<MentorResponse> {
    const response = await apiClient.post('/mentor/ask', request);
    return response.data;
  }

  // ML Models (Direct Endpoints)
  async predictDirection(request: MLPredictionRequest): Promise<DirectionPrediction> {
    const response = await apiClient.post('/ml/predict/direction', request);
    return response.data;
  }

  async forecastVolatility(request: MLPredictionRequest): Promise<VolatilityForecast> {
    const response = await apiClient.post('/ml/forecast/volatility', request);
    return response.data;
  }

  async classifyRegime(request: MLPredictionRequest): Promise<RegimeClassification> {
    const response = await apiClient.post('/ml/classify/regime', request);
    return response.data;
  }

  async listMLModels(): Promise<{ models: Array<{ name: string; type: string; status: string }> }> {
    const response = await apiClient.get('/ml/models/list');
    return response.data;
  }

  // Configuration
  async getConfig(): Promise<Record<string, any>> {
    const response = await apiClient.get('/config');
    return response.data;
  }
}

// Export singleton instance
export const api = new ApiService();

// Export client for custom requests
export { apiClient };

// Helper function to check backend connectivity
export async function checkBackendConnection(): Promise<boolean> {
  try {
    await api.health();
    return true;
  } catch (error) {
    console.error('Backend not reachable:', error);
    return false;
  }
}
