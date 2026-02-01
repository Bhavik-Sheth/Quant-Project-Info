"""
CF-AI-SDE Unified Backend Integration Pipeline
==============================================

This module provides a single entry point for all backend functionality.
It integrates Data-inges-fe, ML_Models, AI_Agents, quant_strategy, 
Backtesting_risk, RAG_Mentor, and db modules into a cohesive API.

Usage:
    from logical_pipe import TradingSystemAPI
    
    api = TradingSystemAPI("config.yaml")
    results = api.run_full_pipeline(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy_name="ml_enhanced"
    )

Author: CF-AI-SDE Team
Version: 1.0.0
"""

import os
import sys
import yaml
import logging
import pickle
import glob
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Callable, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from tinydb import TinyDB, Query

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Management
# =============================================================================

class ConfigLoader:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Resolve relative paths to absolute, relative to this file's directory
        if not os.path.isabs(config_path):
            # Get the directory where logical_pipe.py is located (backend/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not os.path.exists(self.config_path):
            # Provide detailed error message
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Expected config at: {os.path.abspath(self.config_path)}"
            )
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_env_vars(self) -> None:
        """Replace ${VAR_NAME} with environment variables"""
        def resolve_dict(d: Dict) -> Dict:
            for key, value in d.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    d[key] = os.getenv(env_var, "")
                elif isinstance(value, dict):
                    d[key] = resolve_dict(value)
            return d
        
        self.config = resolve_dict(self.config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'ml_models.direction.default_model')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """Get entire config section"""
        return self.config.get(section, {})
    
    def validate(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ["system", "data_ingestion", "ml_models", "ai_agents", "strategies", "risk_management"]
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        return True


# =============================================================================
# Temporal Integrity Coordinator
# =============================================================================

class TemporalCoordinator:
    """Ensures no look-ahead bias throughout pipeline"""
    
    def __init__(self):
        self.current_timestamp = None
        self.context_at_t = {}
        self.execution_at_t_plus_1 = []
        self.audit_log = []
    
    def advance_time(self, timestamp: datetime) -> None:
        """Move to next timestamp, validate temporal integrity"""
        if self.current_timestamp and timestamp <= self.current_timestamp:
            raise ValueError(f"Time cannot move backward: {timestamp} <= {self.current_timestamp}")
        self.current_timestamp = timestamp
        self.audit_log.append({"timestamp": timestamp, "action": "time_advanced"})
    
    def validate_context(self, context: Dict) -> bool:
        """Ensure all data in context is <= current_timestamp"""
        for key, value in context.items():
            if isinstance(value, pd.DataFrame) and 'timestamp' in value.columns:
                max_time = value['timestamp'].max()
                if pd.notna(max_time) and max_time > self.current_timestamp:
                    raise ValueError(f"Look-ahead bias detected in {key}: {max_time} > {self.current_timestamp}")
        return True
    
    def execute_pipeline_step(self, step_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute pipeline step with temporal validation"""
        self.audit_log.append({
            "timestamp": self.current_timestamp,
            "step": step_name,
            "status": "started"
        })
        
        try:
            result = func(*args, **kwargs)
            self.audit_log.append({
                "timestamp": self.current_timestamp,
                "step": step_name,
                "status": "completed"
            })
            return result
        except Exception as e:
            self.audit_log.append({
                "timestamp": self.current_timestamp,
                "step": step_name,
                "status": "failed",
                "error": str(e)
            })
            raise
    
    def get_audit_trail(self) -> List[Dict]:
        """Return complete audit log"""
        return self.audit_log


# =============================================================================
# Data Pipeline (Wraps Data-inges-fe module)
# =============================================================================

class DataPipeline:
    """Wraps Data-inges-fe module for data ingestion and feature engineering"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._setup_data_directories()
    
    def _setup_data_directories(self):
        """Create necessary data directories"""
        for dir_name in ['data/raw', 'data/validated', 'data/features', 'data/fallback']:
            os.makedirs(dir_name, exist_ok=True)
    
    def ingest_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Ingest OHLCV data from Yahoo Finance
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe (1d, 1h, etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use importlib for hyphenated module name
            equity_ohlcv_module = importlib.import_module('Data-inges-fe.src.ingestion.equity_ohlcv')
            EquityOHLCVFetcher = equity_ohlcv_module.EquityOHLCVFetcher
            
            logger.info(f"Ingesting data for {symbols} from {start_date} to {end_date}")
            
            fetcher = EquityOHLCVFetcher()
            all_data = []
            
            for symbol in symbols:
                try:
                    data = fetcher.fetch_ohlcv(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=timeframe
                    )
                    if data is not None and not data.empty:
                        data['symbol'] = symbol
                        all_data.append(data)
                        logger.info(f"Fetched {len(data)} records for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            if not all_data:
                raise ValueError("No data fetched for any symbol")
            
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records ingested: {len(combined_data)}")
            
            return combined_data
            
        except ImportError as e:
            logger.error(f"Failed to import Data-inges-fe module: {e}")
            raise
    
    def engineer_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer technical indicators from raw OHLCV data
        
        Args:
            raw_data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with 70+ technical indicators
        """
        try:
            # Use importlib for hyphenated module name
            tech_indicators_module = importlib.import_module('Data-inges-fe.src.features.technical_indicators')
            normalization_module = importlib.import_module('Data-inges-fe.src.features.normalization')
            TechnicalIndicators = tech_indicators_module.TechnicalIndicators
            FeatureNormalizer = normalization_module.FeatureNormalizer
            
            logger.info("Engineering technical features")
            
            ti = TechnicalIndicators()
            features = ti.compute_all_indicators(raw_data)
            
            logger.info(f"Generated {len(features.columns)} features")
            
            # Normalize features
            normalizer = FeatureNormalizer()
            normalized_features = normalizer.normalize(features)
            
            logger.info("Feature engineering completed")
            return normalized_features
            
        except ImportError as e:
            logger.error(f"Failed to import feature engineering modules: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Tuple of (validated_data, validation_report)
        """
        try:
            # Use importlib for hyphenated module name
            ohlcv_checks_module = importlib.import_module('Data-inges-fe.src.validation.ohlcv_checks')
            OHLCVValidator = ohlcv_checks_module.OHLCVValidator
            
            logger.info("Validating data quality")
            
            validator = OHLCVValidator()
            validation_report = validator.validate(data)
            
            # Remove invalid records if specified
            if validation_report.get('has_errors', False):
                logger.warning(f"Found {validation_report.get('error_count', 0)} validation errors")
                # Keep only valid records
                valid_data = data[~data.index.isin(validation_report.get('invalid_indices', []))]
            else:
                valid_data = data
                logger.info("All data passed validation")
            
            return valid_data, validation_report
            
        except ImportError as e:
            logger.warning(f"Validation module not available: {e}. Skipping validation.")
            return data, {'validation_skipped': True}
    
    def get_latest_data(self, symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        """
        Get latest data for a symbol
        
        Args:
            symbol: Ticker symbol
            lookback_days: Number of days to look back
        
        Returns:
            DataFrame with recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.ingest_data(
            symbols=[symbol],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Check if data pipeline is operational"""
        status = {'status': 'healthy', 'issues': []}
        
        # Check data directories
        for dir_name in ['data/raw', 'data/validated', 'data/features']:
            if not os.path.exists(dir_name):
                status['status'] = 'degraded'
                status['issues'].append(f"Missing directory: {dir_name}")
        
        # Try importing modules
        try:
            # Use importlib for hyphenated module name
            importlib.import_module('Data-inges-fe.src.ingestion.equity_ohlcv')
            importlib.import_module('Data-inges-fe.src.features.technical_indicators')
        except ImportError as e:
            status['status'] = 'unhealthy'
            status['issues'].append(f"Module import failed: {e}")
        
        return status


# =============================================================================
# Model Manager (Wraps ML_Models module)
# =============================================================================

class ModelManager:
    """Wraps ML_Models module with persistence"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.persistence_enabled = config.get('persistence', {}).get('enabled', True)
        self.storage = config.get('persistence', {}).get('storage', 'mongodb')
        self.models = {}  # Cache for loaded models
        
        # Setup model storage directory
        os.makedirs('models', exist_ok=True)
        
        # Initialize db writers/readers if enabled
        if self.persistence_enabled and self.storage == 'mongodb':
            try:
                from db.connection import get_mongodb_client
                from db.writers import MLModelWriter
                from db.readers import MLModelReader
                
                client = get_mongodb_client()
                if client is not None:
                    self.model_writer = MLModelWriter(client)
                    self.model_reader = MLModelReader(client)
                    self.db_available = True
                    logger.info("MongoDB persistence enabled for models")
                else:
                    logger.warning("MongoDB unavailable, using file-based storage")
                    self.db_available = False
            except Exception as e:
                logger.warning(f"MongoDB unavailable, using file-based storage: {e}")
                self.db_available = False
        else:
            self.db_available = False
    
    def train_direction_model(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        model_type: str = "xgboost"
    ) -> Dict:
        """
        Train direction prediction model
        
        Args:
            features: Feature DataFrame
            targets: Binary target (0=Down, 1=Up)
            model_type: Model type (xgboost, lstm, logistic, etc.)
        
        Returns:
            Dict with model, metrics, and model_id
        """
        try:
            from ML_Models.direction_pred import (
                XGBoostDirectionPredictor,
                LSTMDirectionPredictor
            )
            
            logger.info(f"Training {model_type} direction model")
            
            if model_type == "xgboost":
                model = XGBoostDirectionPredictor()
            elif model_type == "lstm":
                model = LSTMDirectionPredictor()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            model.train(features, targets)
            
            # Get metrics
            predictions = model.predict(features)
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {
                'accuracy': accuracy_score(targets, predictions),
                'f1_score': f1_score(targets, predictions, average='weighted')
            }
            
            # Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = self.save_model(model, 'direction', version, metrics)
            
            logger.info(f"Direction model trained. Accuracy: {metrics['accuracy']:.3f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'model_id': model_id,
                'model_type': model_type
            }
            
        except ImportError as e:
            logger.error(f"Failed to import direction prediction module: {e}")
            raise
    
    def train_volatility_model(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        model_type: str = "garch"
    ) -> Dict:
        """
        Train volatility forecasting model
        
        Args:
            features: Feature DataFrame
            targets: Volatility targets
            model_type: Model type (garch, egarch, lstm)
        
        Returns:
            Dict with model, metrics, and model_id
        """
        try:
            from ML_Models.Volatility_Forecasting import VolatilityGARCH, VolatilityLSTM
            
            logger.info(f"Training {model_type} volatility model")
            
            if model_type in ["garch", "egarch"]:
                model = VolatilityGARCH()
                model.garch_train(targets, model_type=model_type)
            elif model_type == "lstm":
                model = VolatilityLSTM()
                model.train(features, targets)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get metrics (simplified)
            metrics = {'model_type': model_type, 'trained_at': datetime.now().isoformat()}
            
            # Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = self.save_model(model, 'volatility', version, metrics)
            
            logger.info(f"Volatility model trained and saved")
            
            return {
                'model': model,
                'metrics': metrics,
                'model_id': model_id,
                'model_type': model_type
            }
            
        except ImportError as e:
            logger.error(f"Failed to import volatility forecasting module: {e}")
            raise
    
    def train_regime_model(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        model_type: str = "random_forest"
    ) -> Dict:
        """
        Train regime classification model
        
        Args:
            features: Feature DataFrame
            targets: Regime labels
            model_type: Model type (random_forest, lstm)
        
        Returns:
            Dict with model, metrics, and model_id
        """
        try:
            from ML_Models.Regime_Classificaiton import RegimeClassifier
            
            logger.info(f"Training {model_type} regime model")
            
            classifier = RegimeClassifier()
            
            if model_type == "random_forest":
                classifier.train_random_forest(features, targets)
            elif model_type == "lstm":
                classifier.train_lstm(features, targets)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get metrics
            predictions = classifier.predict(features)
            from sklearn.metrics import accuracy_score
            metrics = {
                'accuracy': accuracy_score(targets, predictions)
            }
            
            # Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = self.save_model(classifier, 'regime', version, metrics)
            
            logger.info(f"Regime model trained. Accuracy: {metrics['accuracy']:.3f}")
            
            return {
                'model': classifier,
                'metrics': metrics,
                'model_id': model_id,
                'model_type': model_type
            }
            
        except ImportError as e:
            logger.error(f"Failed to import regime classification module: {e}")
            raise
    
    def train_gan(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Train GAN for synthetic data generation
        
        Args:
            features: Feature array (num_samples, seq_len, num_features)
            labels: Label array
        
        Returns:
            Dict with model and metrics
        """
        try:
            from ML_Models.GAN import MarketGAN
            
            logger.info("Training GAN model")
            
            gan = MarketGAN(
                seq_len=features.shape[1],
                num_features=features.shape[2],
                num_classes=len(np.unique(labels))
            )
            
            gan.train(features, labels, epochs=100)
            
            # Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = self.save_model(gan, 'gan', version, {'trained_at': datetime.now().isoformat()})
            
            logger.info("GAN model trained and saved")
            
            return {
                'model': gan,
                'metrics': {},
                'model_id': model_id,
                'model_type': 'gan'
            }
            
        except ImportError as e:
            logger.error(f"Failed to import GAN module: {e}")
            raise
    
    def predict_direction(self, features: pd.DataFrame, model_name: str = "latest") -> np.ndarray:
        """Predict price direction"""
        try:
            model = self.load_model('direction', model_name)
            predictions = model.predict(features)
            return predictions
        except Exception as e:
            logger.error(f"Direction prediction failed: {e}")
            raise
    
    def predict_volatility(self, features: pd.DataFrame, model_name: str = "latest") -> np.ndarray:
        """Predict volatility"""
        try:
            model = self.load_model('volatility', model_name)
            if hasattr(model, 'forecast'):
                forecast = model.forecast(steps=5)
                return forecast['volatility']
            else:
                return model.predict(features)
        except Exception as e:
            logger.error(f"Volatility prediction failed: {e}")
            raise
    
    def predict_regime(self, features: pd.DataFrame, model_name: str = "latest") -> str:
        """Predict market regime"""
        try:
            model = self.load_model('regime', model_name)
            prediction = model.predict(features)
            return prediction
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            raise
    
    def save_model(self, model: Any, model_type: str, version: str, metrics: Dict) -> str:
        """Save model with versioning"""
        if self.db_available:
            try:
                # Serialize model
                model_data = pickle.dumps(model)
                
                metadata = {
                    'version': version,
                    'metrics': metrics,
                    'hyperparameters': getattr(model, 'get_params', lambda: {})(),
                    'framework': type(model).__module__.split('.')[0]
                }
                
                model_id = self.model_writer.save_model(model_type, model_data, metadata)
                logger.info(f"Model saved to MongoDB: {model_id}")
                
                # Cache model
                self.models[f"{model_type}_{version}"] = model
                
                return model_id
            except Exception as e:
                logger.warning(f"MongoDB save failed: {e}. Falling back to file storage.")
                self.db_available = False
        
        # Fallback: Save to file system
        filepath = f"models/{model_type}_{version}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to file: {filepath}")
        
        # Cache model
        self.models[f"{model_type}_{version}"] = model
        
        return filepath
    
    def load_model(self, model_type: str, version: str = 'latest') -> Any:
        """Load model from storage"""
        # Check cache first
        cache_key = f"{model_type}_{version}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        if self.db_available:
            try:
                model, metadata = self.model_reader.load_model(model_type, version)
                logger.info(f"Model loaded from MongoDB: {metadata['model_id']}")
                
                # Cache model
                self.models[cache_key] = model
                
                return model
            except Exception as e:
                logger.warning(f"MongoDB load failed: {e}. Trying file storage.")
                self.db_available = False
        
        # Fallback: Load from file system
        if version == 'latest':
            # Find latest file
            files = glob.glob(f"models/{model_type}_*.pkl")
            if not files:
                raise FileNotFoundError(f"No saved model for {model_type}")
            filepath = max(files, key=os.path.getctime)
        else:
            filepath = f"models/{model_type}_{version}.pkl"
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from file: {filepath}")
        
        # Cache model
        self.models[cache_key] = model
        
        return model
    
    def list_models(self, model_type: str = None) -> List[Dict]:
        """List available models"""
        if self.db_available:
            try:
                return self.model_reader.list_models(model_type)
            except:
                pass
        
        # Fallback: List files
        pattern = f"models/{model_type}_*.pkl" if model_type else "models/*.pkl"
        files = glob.glob(pattern)
        return [
            {
                'filepath': f,
                'model_type': Path(f).stem.split('_')[0],
                'size_mb': os.path.getsize(f) / (1024*1024)
            }
            for f in files
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check if model manager is operational"""
        status = {'status': 'healthy', 'issues': []}
        
        # Check models directory
        if not os.path.exists('models'):
            status['status'] = 'degraded'
            status['issues'].append("Missing models directory")
        
        # Check persistence
        if self.persistence_enabled and not self.db_available:
            status['status'] = 'degraded'
            status['issues'].append("MongoDB persistence unavailable, using file storage")
        
        return status


# =============================================================================
# Agent Orchestrator (Wraps AI_Agents module)
# =============================================================================

class AgentOrchestrator:
    """Wraps AI_Agents module with persistent performance tracking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {}
        
        # Initialize db writers/readers
        try:
            from db.connection import get_mongodb_client
            from db.writers import AgentMemoryWriter
            from db.readers import AgentMemoryReader
            
            client = get_mongodb_client()
            if client is not None:
                self.memory_writer = AgentMemoryWriter(client)
                self.memory_reader = AgentMemoryReader(client)
                self.db_available = True
                logger.info("MongoDB persistence enabled for agents")
            else:
                logger.warning("MongoDB unavailable for agent memory")
                self.db_available = False
        except Exception as e:
            logger.warning(f"MongoDB unavailable for agent memory: {e}")
            self.db_available = False
        
        self.initialize_agents()
    
    def initialize_agents(self) -> None:
        """Initialize agents and restore performance weights from DB"""
        try:
            from AI_Agents.agents import (
                MarketDataAgent,
                RiskMonitoringAgent,
                SentimentAgent,
                VolatilityAgent,
                RegimeDetectionAgent,
                SignalAggregatorAgent
            )
            
            enabled = self.config.get('enabled_agents', [])
            
            agent_classes = {
                'market_data': MarketDataAgent,
                'risk_monitoring': RiskMonitoringAgent,
                'sentiment': SentimentAgent,
                'volatility': VolatilityAgent,
                'regime': RegimeDetectionAgent,
                'signal_aggregator': SignalAggregatorAgent
            }
            
            for agent_name, agent_class in agent_classes.items():
                if agent_name in enabled:
                    try:
                        agent = agent_class(name=agent_name)
                        
                        # Restore performance weight from database
                        if self.db_available:
                            try:
                                saved_weight = self.memory_reader.get_latest_weight(agent_name)
                                agent.performance_weight = saved_weight
                                logger.info(f"Restored {agent_name} weight: {saved_weight:.3f}")
                            except Exception as e:
                                logger.warning(f"Could not restore weight for {agent_name}: {e}")
                        
                        self.agents[agent_name] = agent
                        logger.info(f"Initialized agent: {agent_name}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {agent_name}: {e}")
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except ImportError as e:
            logger.error(f"Failed to import AI_Agents module: {e}")
            raise
    
    def run_all_agents(self, context: Dict) -> List:
        """
        Run all enabled agents in parallel
        
        Args:
            context: Dict with market data, features, predictions, etc.
        
        Returns:
            List of AgentResponse objects
        """
        logger.info(f"Running {len(self.agents)} agents")
        
        agent_responses = []
        
        for agent_name, agent in self.agents.items():
            if agent_name == 'signal_aggregator':
                continue  # Aggregator runs last
            
            try:
                response = agent.analyze(context)
                agent_responses.append(response)
                logger.info(f"{agent_name} completed: {response.action}")
            except Exception as e:
                logger.error(f"{agent_name} failed: {e}")
        
        return agent_responses
    
    def run_single_agent(self, agent_name: str, context: Dict):
        """Run a single agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        return agent.analyze(context)
    
    def get_aggregated_signal(self, agent_responses: List):
        """
        Get aggregated signal from SignalAggregatorAgent
        
        Args:
            agent_responses: List of responses from other agents
        
        Returns:
            Final aggregated signal
        """
        if 'signal_aggregator' not in self.agents:
            logger.warning("SignalAggregatorAgent not enabled, using simple majority voting")
            # Simple majority voting fallback
            actions = [r.action for r in agent_responses]
            from collections import Counter
            most_common = Counter(actions).most_common(1)[0][0]
            return {'action': most_common, 'confidence': 0.5, 'reason': 'Majority vote'}
        
        aggregator = self.agents['signal_aggregator']
        
        # Create context with agent responses
        context = {'agent_responses': agent_responses}
        
        final_signal = aggregator.analyze(context)
        logger.info(f"Aggregated signal: {final_signal.action} (confidence: {final_signal.confidence:.2f})")
        
        return final_signal
    
    def update_agent_performance(self, agent_name: str, actual_outcome: float) -> None:
        """Update agent performance and persist to database"""
        if agent_name not in self.agents:
            return
        
        agent = self.agents[agent_name]
        
        # Update agent's internal performance tracking
        if hasattr(agent, 'update_performance'):
            agent.update_performance(actual_outcome)
        
        # Persist to database
        if self.db_available:
            try:
                metadata = {
                    'accuracy': getattr(agent, 'accuracy', 0.0),
                    'total_predictions': getattr(agent, 'total_predictions', 0),
                    'correct_predictions': getattr(agent, 'correct_predictions', 0),
                    'session_id': self.config.get('session_id', 'default')
                }
                
                self.memory_writer.save_agent_weights(
                    agent_name,
                    agent.performance_weight,
                    metadata
                )
                logger.info(f"Persisted {agent_name} weight: {agent.performance_weight:.3f}")
            except Exception as e:
                logger.error(f"Failed to persist agent weight: {e}")
    
    def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                'enabled': True,
                'performance_weight': getattr(agent, 'performance_weight', 1.0),
                'total_predictions': getattr(agent, 'total_predictions', 0),
                'accuracy': getattr(agent, 'accuracy', 0.0)
            }
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Check if agent orchestrator is operational"""
        status = {'status': 'healthy', 'issues': []}
        
        if not self.agents:
            status['status'] = 'unhealthy'
            status['issues'].append("No agents initialized")
        
        # Check each agent
        for agent_name, agent in self.agents.items():
            if not hasattr(agent, 'analyze'):
                status['status'] = 'degraded'
                status['issues'].append(f"{agent_name} missing analyze method")
        
        return status


# =============================================================================
# Strategy Engine (Unified Backtesting)
# =============================================================================

class StrategyEngine:
    """Unified backtesting engine (wraps quant_strategy and Backtesting_risk)"""
    
    def __init__(self, strategy_config: Dict, risk_config: Dict):
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.strategies = {}
    
    def register_strategy(self, strategy_name: str, strategy_obj: Any) -> None:
        """Register a strategy for backtesting"""
        self.strategies[strategy_name] = strategy_obj
        logger.info(f"Registered strategy: {strategy_name}")
    
    def backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strategies: List[str],
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run backtest using quant_strategy engine
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            strategies: List of strategy names
            initial_capital: Starting capital
        
        Returns:
            Dict with metrics, trades, and equity curve
        """
        try:
            from quant_strategy.engine import BacktestEngine
            from quant_strategy.strategies.technical import RSIStrategy, MAStrategy
            from quant_strategy.strategies.ml_enhanced import MLEnhancedStrategy
            
            logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
            
            # Create backtest engine
            engine = BacktestEngine(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            # Add strategies
            for strategy_name in strategies:
                if strategy_name == 'rsi':
                    strategy = RSIStrategy()
                elif strategy_name == 'ma_cross':
                    strategy = MAStrategy()
                elif strategy_name == 'ml_enhanced':
                    strategy = MLEnhancedStrategy()
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
                    continue
                
                engine.add_strategy(strategy)
            
            # Run backtest
            results = engine.run()
            
            logger.info(f"Backtest completed. Total return: {results['metrics']['total_return']:.2%}")
            
            return results
            
        except ImportError as e:
            logger.error(f"Failed to import quant_strategy module: {e}")
            raise
    
    def advanced_backtest(
        self,
        data: pd.DataFrame,
        decisions: List,
        engine: str = "backtesting_risk"
    ) -> Dict:
        """
        Run advanced backtest using Backtesting_risk engine
        
        Args:
            data: Market data DataFrame
            decisions: List of TradeDecision objects
            engine: Engine to use
        
        Returns:
            Performance metrics and trade log
        """
        try:
            from Backtesting_risk.backtesting import BacktestEngine as AdvancedEngine
            from Backtesting_risk.models import ExecutionConfig, RiskConfig
            
            logger.info("Running advanced backtest")
            
            exec_config = ExecutionConfig()
            risk_config = RiskConfig()
            
            engine = AdvancedEngine(data, exec_config, risk_config)
            results = engine.run_backtest(decisions)
            
            logger.info("Advanced backtest completed")
            
            return results
            
        except ImportError as e:
            logger.error(f"Failed to import Backtesting_risk module: {e}")
            raise
    
    def get_performance_metrics(self, trade_log: List[Dict]) -> Dict:
        """Calculate performance metrics from trade log"""
        try:
            from Backtesting_risk.analysis import PerformanceMetrics
            
            metrics = PerformanceMetrics.calculate(trade_log)
            return metrics
            
        except ImportError:
            # Simple metrics calculation
            if not trade_log:
                return {}
            
            total_pnl = sum(t.get('pnl', 0) for t in trade_log)
            winning_trades = [t for t in trade_log if t.get('pnl', 0) > 0]
            
            return {
                'total_trades': len(trade_log),
                'total_pnl': total_pnl,
                'win_rate': len(winning_trades) / len(trade_log) if trade_log else 0
            }
    
    def export_results(self, results: Dict, format: str = "csv") -> str:
        """Export backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            filepath = f"results/backtest_{timestamp}.csv"
            os.makedirs('results', exist_ok=True)
            
            # Convert trades to DataFrame and save
            if 'trades' in results:
                df = pd.DataFrame(results['trades'])
                df.to_csv(filepath, index=False)
                logger.info(f"Results exported to {filepath}")
                return filepath
        
        elif format == "json":
            import json
            filepath = f"results/backtest_{timestamp}.json"
            os.makedirs('results', exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results exported to {filepath}")
            return filepath
        
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if strategy engine is operational"""
        status = {'status': 'healthy', 'issues': []}
        
        # Try importing modules
        try:
            from quant_strategy import engine
        except ImportError as e:
            status['status'] = 'degraded'
            status['issues'].append(f"quant_strategy unavailable: {e}")
        
        try:
            from Backtesting_risk import backtesting
        except ImportError as e:
            status['status'] = 'degraded'
            status['issues'].append(f"Backtesting_risk unavailable: {e}")
        
        return status


# =============================================================================
# Analysis Interface (Wraps RAG_Mentor module)
# =============================================================================

class AnalysisInterface:
    """Wraps RAG_Mentor module for backtest analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mentor = None
        
        if config.get('enabled', True):
            try:
                from RAG_Mentor.interface.trading_mentor import TradingMentor
                self.mentor = TradingMentor()
                logger.info("RAG Mentor initialized")
            except ImportError as e:
                logger.warning(f"RAG Mentor unavailable: {e}")
    
    def analyze_backtest(
        self,
        performance: Dict,
        trades: List[Dict],
        symbols: List[str]
    ) -> Dict:
        """
        Analyze backtest results using RAG Mentor
        
        Args:
            performance: Performance metrics dict
            trades: List of trade dicts
            symbols: List of symbols traded
        
        Returns:
            Analysis dict with summary, violations, and suggestions
        """
        if self.mentor is None:
            logger.warning("RAG Mentor not available, returning basic analysis")
            return {
                'summary': "RAG Mentor not available",
                'performance_analysis': performance,
                'violation_report': {},
                'improvement_suggestions': []
            }
        
        try:
            analysis = self.mentor.analyze_performance(performance, trades, symbols)
            logger.info("Backtest analysis completed")
            return analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'summary': f"Analysis failed: {e}",
                'performance_analysis': performance,
                'violation_report': {},
                'improvement_suggestions': []
            }
    
    def ask_question(self, question: str, context: Dict = None) -> str:
        """Ask a question to the RAG Mentor"""
        if self.mentor is None:
            return "RAG Mentor not available"
        
        try:
            response = self.mentor.ask_question(question)
            return response
        except Exception as e:
            logger.error(f"Question failed: {e}")
            return f"Error: {e}"
    
    def detect_violations(self, trades: List[Dict]) -> Dict:
        """Detect trading principle violations"""
        if self.mentor is None:
            return {}
        
        try:
            from RAG_Mentor.mentor.principle_checker import PrincipleChecker
            checker = PrincipleChecker()
            violations = checker.check_violations(trades)
            return violations
        except Exception as e:
            logger.error(f"Violation detection failed: {e}")
            return {}
    
    def suggest_improvements(
        self,
        performance: Dict,
        trades: List[Dict],
        violations: Dict
    ) -> List[str]:
        """Generate improvement suggestions"""
        if self.mentor is None:
            return ["RAG Mentor not available for suggestions"]
        
        try:
            from RAG_Mentor.mentor.improvement_engine import ImprovementEngine
            engine = ImprovementEngine()
            suggestions = engine.generate_suggestions(performance, trades, violations)
            return suggestions
        except Exception as e:
            logger.error(f"Improvement suggestions failed: {e}")
            return [f"Error: {e}"]
    
    def benchmark_comparison(self, performance: Dict, benchmark: str = "SPY") -> Dict:
        """Compare performance to benchmark"""
        if self.mentor is None:
            return {}
        
        try:
            from RAG_Mentor.mentor.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            comparison = analyzer.benchmark_comparison(performance, benchmark)
            return comparison
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if analysis interface is operational"""
        status = {'status': 'healthy', 'issues': []}
        
        if self.mentor is None:
            status['status'] = 'degraded'
            status['issues'].append("RAG Mentor not initialized")
        
        # Check ChromaDB
        chromadb_path = self.config.get('chromadb_path', './RAG_Mentor/chroma_db')
        if not os.path.exists(chromadb_path):
            status['status'] = 'degraded'
            status['issues'].append(f"ChromaDB not found at {chromadb_path}")
        
        return status


# =============================================================================
# Trading System API (Main Facade)
# =============================================================================

class TradingSystemAPI:
    """
    Main API facade for CF-AI-SDE Trading System
    
    Single import for all functionality:
        from logical_pipe import TradingSystemAPI
        
        api = TradingSystemAPI("config.yaml")
        results = api.run_full_pipeline(["AAPL"], "2023-01-01", "2023-12-31", "ml_enhanced")
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trading system with configuration
        
        Args:
            config_path: Path to YAML config file
        """
        logger.info(f"Initializing CF-AI-SDE Trading System v1.0.0")
        
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Validate config
        self.config_loader.validate()
        
        # Initialize all components
        logger.info("Initializing components...")
        
        self.data_pipeline = DataPipeline(self.config.get('data_ingestion', {}))
        self.model_manager = ModelManager(self.config.get('ml_models', {}))
        self.agent_orchestrator = AgentOrchestrator(self.config.get('ai_agents', {}))
        self.strategy_engine = StrategyEngine(
            self.config.get('strategies', {}),
            self.config.get('risk_management', {})
        )
        self.analysis_interface = AnalysisInterface(self.config.get('rag_mentor', {}))
        self.temporal_coordinator = TemporalCoordinator()
        
        logger.info("CF-AI-SDE Trading System initialized successfully")
    
    def run_full_pipeline(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strategy_name: str,
        use_agents: bool = True,
        use_ml_models: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete trading pipeline from data ingestion to analysis
        
        Args:
            symbols: List of ticker symbols (e.g., ["AAPL", "MSFT"])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy_name: Strategy to use (e.g., "ml_enhanced", "rsi", "ma_cross")
            use_agents: Whether to use AI agents for signal generation
            use_ml_models: Whether to use ML models for predictions
        
        Returns:
            Dict with keys:
                - performance: Performance metrics (Sharpe, returns, etc.)
                - trades: List of executed trades
                - agent_outputs: Agent responses (if use_agents=True)
                - ml_predictions: ML model predictions (if use_ml_models=True)
                - analysis: RAG mentor analysis
                - audit_trail: Temporal coordinator log
        
        Example:
            >>> api = TradingSystemAPI()
            >>> results = api.run_full_pipeline(
            ...     symbols=["AAPL"],
            ...     start_date="2023-01-01",
            ...     end_date="2023-12-31",
            ...     strategy_name="ml_enhanced"
            ... )
            >>> print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
            Sharpe Ratio: 1.85
        """
        try:
            logger.info(f"Starting full pipeline for {symbols} from {start_date} to {end_date}")
            
            # Step 1: Data Ingestion
            logger.info("Step 1/7: Data ingestion")
            raw_data = self.data_pipeline.ingest_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.get('data_ingestion', {}).get('default_timeframe', '1d')
            )
            
            # Step 2: Feature Engineering
            logger.info("Step 2/7: Feature engineering")
            features = self.data_pipeline.engineer_features(raw_data)
            
            # Step 3: ML Predictions (optional)
            ml_predictions = {}
            if use_ml_models:
                logger.info("Step 3/7: ML predictions")
                try:
                    ml_predictions = {
                        'direction': self.model_manager.predict_direction(features, 'latest'),
                        'volatility': self.model_manager.predict_volatility(features, 'latest'),
                        'regime': self.model_manager.predict_regime(features, 'latest')
                    }
                except Exception as e:
                    logger.warning(f"ML predictions failed: {e}. Continuing without ML predictions.")
                    ml_predictions = {}
            else:
                logger.info("Step 3/7: ML predictions (skipped)")
            
            # Step 4: Agent Analysis (optional)
            agent_outputs = []
            if use_agents:
                logger.info("Step 4/7: AI agent analysis")
                try:
                    context = {
                        'data': raw_data,
                        'features': features,
                        'ml_predictions': ml_predictions,
                        'symbols': symbols
                    }
                    agent_outputs = self.agent_orchestrator.run_all_agents(context)
                except Exception as e:
                    logger.warning(f"Agent analysis failed: {e}. Continuing without agents.")
                    agent_outputs = []
            else:
                logger.info("Step 4/7: AI agent analysis (skipped)")
            
            # Step 5: Backtesting
            logger.info("Step 5/7: Strategy backtesting")
            backtest_results = self.strategy_engine.backtest(
                symbol=symbols[0],  # TODO: Support multiple symbols
                start_date=start_date,
                end_date=end_date,
                strategies=[strategy_name]
            )
            
            # Step 6: RAG Analysis
            logger.info("Step 6/7: Performance analysis")
            analysis = self.analysis_interface.analyze_backtest(
                performance=backtest_results.get('metrics', {}),
                trades=backtest_results.get('trades', []),
                symbols=symbols
            )
            
            # Step 7: Compile results
            logger.info("Step 7/7: Compiling results")
            results = {
                'status': 'success',
                'performance': backtest_results.get('metrics', {}),
                'trades': backtest_results.get('trades', []),
                'agent_outputs': agent_outputs,
                'ml_predictions': ml_predictions,
                'analysis': analysis,
                'audit_trail': self.temporal_coordinator.get_audit_trail(),
                'metadata': {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'strategy': strategy_name,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            sharpe = results['performance'].get('sharpe_ratio', 0)
            logger.info(f"Pipeline completed successfully. Sharpe: {sharpe:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'audit_trail': self.temporal_coordinator.get_audit_trail()
            }
    
    def run_partial_pipeline(self, stage: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specific pipeline stage only
        
        Args:
            stage: One of ['ingest', 'features', 'ml', 'agents', 'backtest', 'analyze']
            inputs: Stage-specific inputs
        
        Returns:
            Dict with stage-specific outputs
        
        Example:
            >>> # Just ingest data
            >>> data = api.run_partial_pipeline('ingest', {
            ...     'symbols': ['AAPL'],
            ...     'start_date': '2023-01-01',
            ...     'end_date': '2023-12-31'
            ... })
            
            >>> # Just run agents
            >>> agent_outputs = api.run_partial_pipeline('agents', {
            ...     'context': {'data': data, 'features': features}
            ... })
        """
        try:
            if stage == 'ingest':
                return self.data_pipeline.ingest_data(**inputs)
            
            elif stage == 'features':
                return self.data_pipeline.engineer_features(inputs['data'])
            
            elif stage == 'ml':
                model_type = inputs.get('model_type', 'direction')
                if model_type == 'direction':
                    return self.model_manager.predict_direction(inputs['features'], inputs.get('model_name', 'latest'))
                elif model_type == 'volatility':
                    return self.model_manager.predict_volatility(inputs['features'], inputs.get('model_name', 'latest'))
                elif model_type == 'regime':
                    return self.model_manager.predict_regime(inputs['features'], inputs.get('model_name', 'latest'))
            
            elif stage == 'agents':
                return self.agent_orchestrator.run_all_agents(inputs['context'])
            
            elif stage == 'backtest':
                return self.strategy_engine.backtest(**inputs)
            
            elif stage == 'analyze':
                return self.analysis_interface.analyze_backtest(**inputs)
            
            else:
                raise ValueError(f"Unknown stage: {stage}. Must be one of ['ingest', 'features', 'ml', 'agents', 'backtest', 'analyze']")
        
        except Exception as e:
            logger.error(f"Partial pipeline failed at stage '{stage}': {str(e)}", exc_info=True)
            return {'status': 'error', 'stage': stage, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dict with component statuses and system health
        
        Example:
            >>> status = api.get_system_status()
            >>> print(f"Overall Health: {status['overall_health']}")
            Overall Health: healthy
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_health': 'healthy'
        }
        
        # Check each component
        components = [
            ('data_pipeline', self.data_pipeline),
            ('model_manager', self.model_manager),
            ('agent_orchestrator', self.agent_orchestrator),
            ('strategy_engine', self.strategy_engine),
            ('analysis_interface', self.analysis_interface)
        ]
        
        for name, component in components:
            try:
                # Each component should have a health_check() method
                if hasattr(component, 'health_check'):
                    health = component.health_check()
                else:
                    health = {'status': 'unknown', 'message': 'No health check implemented'}
                
                status['components'][name] = health
                
                if health.get('status') != 'healthy':
                    status['overall_health'] = 'degraded'
            
            except Exception as e:
                status['components'][name] = {'status': 'error', 'error': str(e)}
                status['overall_health'] = 'unhealthy'
        
        return status
    
    def ingest_market_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Fetch and store OHLCV market data
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            timeframe: Data interval (1d, 1h, etc.)
        
        Returns:
            Dict with status, records count, and storage location
        """
        try:
            logger.info(f"Ingesting data for {symbol} from {start_date} to {end_date}")
            
            # Fetch OHLCV data
            raw_data = self.data_pipeline.ingest_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if raw_data.empty:
                return {
                    "status": "error",
                    "message": f"No data found for {symbol} in specified date range",
                    "records": 0
                }
            
            # Engineer features
            featured_data = self.data_pipeline.engineer_features(raw_data)
            
            # Validate data
            validated_data, validation_report = self.data_pipeline.validate_data(featured_data)
            
            # Store data (MongoDB or TinyDB fallback)
            storage_result = self._store_market_data(symbol, validated_data)
            
            return {
                "status": "success",
                "symbol": symbol,
                "records": len(validated_data),
                "date_range": f"{start_date} to {end_date}",
                "storage": storage_result["storage"],
                "validation": validation_report,
                "features_count": len(validated_data.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest data for {symbol}: {e}")
            return {
                "status": "error",
                "message": f"Data ingestion failed: {str(e)}",
                "symbol": symbol
            }
    
    def get_market_data(
        self, 
        symbol: str, 
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve market data from storage
        
        Args:
            symbol: Ticker symbol
            limit: Maximum number of records
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            List of market data records
        """
        try:
            # Try MongoDB first
            try:
                from db.readers import MarketDataReader
                reader = MarketDataReader()
                
                if start_date and end_date:
                    data = reader.get_ohlcv_range(symbol, start_date, end_date)
                else:
                    data = reader.get_latest_ohlcv(symbol, limit)
                
                logger.info(f"Retrieved {len(data)} records from MongoDB for {symbol}")
                return data
                
            except Exception as mongo_error:
                logger.warning(f"MongoDB retrieval failed: {mongo_error}, trying TinyDB fallback")
                
                # Fallback to TinyDB
                db_path = os.path.join('data', 'fallback', 'market_data.json')
                
                if not os.path.exists(db_path):
                    logger.error(f"No data found for {symbol} in TinyDB")
                    return []
                
                db = TinyDB(db_path)
                MarketData = Query()
                
                if start_date and end_date:
                    results = db.search(
                        (MarketData.symbol == symbol) & 
                        (MarketData.timestamp >= start_date) & 
                        (MarketData.timestamp <= end_date)
                    )
                else:
                    results = db.search(MarketData.symbol == symbol)
                    results = sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
                
                db.close()
                logger.info(f"Retrieved {len(results)} records from TinyDB for {symbol}")
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve data for {symbol}: {e}")
            return []
    
    def generate_signals(
        self,
        symbol: str,
        strategy: str = "rsi",
        lookback_period: int = 100
    ) -> Dict[str, Any]:
        """
        Generate trading signals using specified strategy
        
        Args:
            symbol: Ticker symbol
            strategy: Strategy name (rsi, macd, ml_enhanced, multi_agent)
            lookback_period: Number of periods to analyze
        
        Returns:
            Dict with signals and analysis
        """
        try:
            logger.info(f"Generating {strategy} signals for {symbol}")
            
            # Get market data
            data = self.get_market_data(symbol, limit=lookback_period)
            
            if not data:
                return {
                    "status": "error",
                    "message": f"No data available for {symbol}. Please ingest data first.",
                    "symbol": symbol
                }
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            
            # Generate signals based on strategy
            if strategy == "multi_agent":
                # Use AI agents for signal generation
                context = {
                    "symbol": symbol,
                    "data": df,
                    "lookback_period": lookback_period
                }
                agent_responses = self.agent_orchestrator.run_all_agents(context)
                signals = self.agent_orchestrator.get_aggregated_signal(agent_responses)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "strategy": strategy,
                    "signals": signals,
                    "agent_count": len(agent_responses),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Use strategy engine
                signals = self._generate_strategy_signals(df, strategy)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "strategy": strategy,
                    "signals": signals,
                    "data_points": len(df),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return {
                "status": "error",
                "message": f"Signal generation failed: {str(e)}",
                "symbol": symbol,
                "strategy": strategy
            }
    
    def run_backtest(
        self,
        symbol: str,
        strategy: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        
        Args:
            symbol: Ticker symbol
            strategy: Strategy name
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            config: Optional strategy configuration
        
        Returns:
            Dict with backtest results and performance metrics
        """
        try:
            logger.info(f"Running backtest for {symbol} with {strategy} strategy")
            
            # Use existing backtest functionality
            results = self.strategy_engine.backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategies=[strategy],
                initial_capital=initial_capital
            )
            
            # Store backtest results
            self._store_backtest_results(symbol, strategy, results)
            
            return {
                "status": "success",
                "symbol": symbol,
                "strategy": strategy,
                "performance": results.get("performance", {}),
                "trades": results.get("trades", []),
                "equity_curve": results.get("equity_curve", []),
                "date_range": f"{start_date} to {end_date}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return {
                "status": "error",
                "message": f"Backtest failed: {str(e)}",
                "symbol": symbol,
                "strategy": strategy
            }
    
    def run_agent_analysis(
        self,
        symbol: str,
        agent_type: str = "full"
    ) -> Dict[str, Any]:
        """
        Run AI agent analysis on a symbol
        
        Args:
            symbol: Ticker symbol
            agent_type: Type of analysis (market_data, risk, sentiment, volatility, regime, full)
        
        Returns:
            Dict with agent analysis results
        """
        try:
            logger.info(f"Running {agent_type} agent analysis for {symbol}")
            
            # Get latest market data
            data = self.get_market_data(symbol, limit=365)
            
            if not data:
                return {
                    "status": "error",
                    "message": f"No data available for {symbol}",
                    "symbol": symbol
                }
            
            # Prepare context for agents
            context = {
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run agents based on type
            if agent_type == "full":
                responses = self.agent_orchestrator.run_all_agents(context)
                aggregated = self.agent_orchestrator.get_aggregated_signal(responses)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "agent_type": "full",
                    "individual_analyses": responses,
                    "aggregated_signal": aggregated,
                    "agent_count": len(responses),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Run single agent
                response = self.agent_orchestrator.run_single_agent(agent_type, context)
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "agent_type": agent_type,
                    "analysis": response,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Agent analysis failed for {symbol}: {e}")
            return {
                "status": "error",
                "message": f"Agent analysis failed: {str(e)}",
                "symbol": symbol,
                "agent_type": agent_type
            }
    
    def query_rag_mentor(
        self,
        question: str,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query RAG trading mentor for guidance
        
        Args:
            question: User's question
            symbol: Optional symbol for context
            context: Optional additional context
        
        Returns:
            Mentor's response string
        """
        try:
            logger.info(f"Querying RAG mentor: {question[:50]}...")
            
            # Build context
            mentor_context = context or {}
            
            if symbol:
                # Add recent market data for context
                recent_data = self.get_market_data(symbol, limit=30)
                mentor_context["symbol"] = symbol
                mentor_context["recent_data"] = recent_data
            
            mentor_context["question"] = question
            
            # Query analysis interface
            response = self.analysis_interface.ask_question(question, mentor_context)
            
            logger.info("RAG mentor query successful")
            return response
            
        except Exception as e:
            logger.error(f"RAG mentor query failed: {e}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing or contact support if the issue persists."
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current system configuration (sanitized)
        
        Returns:
            Configuration dict without sensitive data
        """
        try:
            config = self.config_loader.config.copy()
            
            # Remove sensitive keys
            sensitive_keys = ['api_key', 'secret', 'password', 'token', 'uri']
            
            def sanitize_dict(d: Dict) -> Dict:
                sanitized = {}
                for key, value in d.items():
                    # Skip sensitive keys
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        sanitized[key] = "***REDACTED***"
                    elif isinstance(value, dict):
                        sanitized[key] = sanitize_dict(value)
                    else:
                        sanitized[key] = value
                return sanitized
            
            return sanitize_dict(config)
            
        except Exception as e:
            logger.error(f"Failed to retrieve config: {e}")
            return {"error": str(e)}
    
    def _store_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Store market data in MongoDB or TinyDB fallback"""
        try:
            # Try MongoDB first
            from db.writers import MarketDataWriter
            writer = MarketDataWriter()
            
            records = data.to_dict('records')
            writer.write_ohlcv(symbol, records)
            
            logger.info(f"Stored {len(records)} records in MongoDB for {symbol}")
            return {"storage": "mongodb", "records": len(records)}
            
        except Exception as mongo_error:
            logger.warning(f"MongoDB storage failed: {mongo_error}, using TinyDB fallback")
            
            # Fallback to TinyDB
            try:
                db_dir = os.path.join('data', 'fallback')
                os.makedirs(db_dir, exist_ok=True)
                
                db_path = os.path.join(db_dir, 'market_data.json')
                db = TinyDB(db_path)
                
                records = data.to_dict('records')
                for record in records:
                    record['symbol'] = symbol
                    record['stored_at'] = datetime.now().isoformat()
                
                db.insert_multiple(records)
                db.close()
                
                logger.info(f"Stored {len(records)} records in TinyDB for {symbol}")
                return {"storage": "tinydb", "records": len(records)}
                
            except Exception as tinydb_error:
                logger.error(f"TinyDB storage also failed: {tinydb_error}")
                return {"storage": "none", "error": str(tinydb_error)}
    
    def _store_backtest_results(self, symbol: str, strategy: str, results: Dict) -> None:
        """Store backtest results in MongoDB or TinyDB fallback"""
        try:
            # Try MongoDB first
            from db.writers import BacktestResultsWriter
            writer = BacktestResultsWriter()
            writer.write_results(symbol, strategy, results)
            logger.info(f"Stored backtest results in MongoDB for {symbol}/{strategy}")
            
        except Exception as mongo_error:
            logger.warning(f"MongoDB storage failed: {mongo_error}, using TinyDB fallback")
            
            try:
                db_dir = os.path.join('data', 'fallback')
                os.makedirs(db_dir, exist_ok=True)
                
                db_path = os.path.join(db_dir, 'backtest_results.json')
                db = TinyDB(db_path)
                
                result_record = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                
                db.insert(result_record)
                db.close()
                
                logger.info(f"Stored backtest results in TinyDB for {symbol}/{strategy}")
                
            except Exception as tinydb_error:
                logger.error(f"Failed to store backtest results: {tinydb_error}")
    
    def _generate_strategy_signals(self, data: pd.DataFrame, strategy: str) -> Dict[str, Any]:
        """Generate signals for basic strategies"""
        signals = {"action": "HOLD", "strength": 0.0, "indicators": {}}
        
        try:
            if strategy == "rsi":
                # RSI strategy
                if 'rsi' in data.columns:
                    latest_rsi = data['rsi'].iloc[-1]
                    signals["indicators"]["rsi"] = float(latest_rsi)
                    
                    if latest_rsi < 30:
                        signals["action"] = "BUY"
                        signals["strength"] = (30 - latest_rsi) / 30
                    elif latest_rsi > 70:
                        signals["action"] = "SELL"
                        signals["strength"] = (latest_rsi - 70) / 30
                        
            elif strategy == "macd":
                # MACD strategy
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    latest_macd = data['macd'].iloc[-1]
                    latest_signal = data['macd_signal'].iloc[-1]
                    
                    signals["indicators"]["macd"] = float(latest_macd)
                    signals["indicators"]["macd_signal"] = float(latest_signal)
                    
                    if latest_macd > latest_signal:
                        signals["action"] = "BUY"
                        signals["strength"] = abs(latest_macd - latest_signal) / abs(latest_signal) if latest_signal != 0 else 0.5
                    elif latest_macd < latest_signal:
                        signals["action"] = "SELL"
                        signals["strength"] = abs(latest_macd - latest_signal) / abs(latest_signal) if latest_signal != 0 else 0.5
                        
            # Add more strategies as needed
            
        except Exception as e:
            logger.error(f"Error generating {strategy} signals: {e}")
            signals["error"] = str(e)
        
        return signals
    
    def health_check(self) -> Dict[str, str]:
        """
        Quick health check for all dependencies
        
        Returns:
            Dict mapping component names to string status ('ok' or 'error')
        
        Example:
            >>> health = api.health_check()
            >>> if all(v == 'ok' for v in health.values()):
            ...     print("All systems operational")
        """
        health = {}
        
        # Config loaded
        health['config_loaded'] = 'ok' if self.config is not None else 'error'
        
        # MongoDB connected (Phase 2)
        try:
            from db.connection import get_mongodb_client
            client = get_mongodb_client()
            health['mongodb_connected'] = 'ok' if client is not None else 'error'
        except:
            health['mongodb_connected'] = 'error'
        
        # LLM APIs ready
        health['gemini_api_ready'] = 'ok' if bool(self.config.get('api_keys', {}).get('gemini')) else 'not_configured'
        health['groq_api_ready'] = 'ok' if bool(self.config.get('api_keys', {}).get('groq')) else 'not_configured'
        
        # ChromaDB initialized
        try:
            import chromadb
            chromadb_path = self.config.get('rag_mentor', {}).get('chromadb_path', './RAG_Mentor/chroma_db')
            health['chromadb_initialized'] = 'ok' if os.path.exists(chromadb_path) else 'error'
        except:
            health['chromadb_initialized'] = 'error'
        
        # Data directories exist
        data_dirs_ok = all([
            os.path.exists('data/raw'),
            os.path.exists('data/validated'),
            os.path.exists('data/features')
        ])
        health['data_directories_created'] = 'ok' if data_dirs_ok else 'error'
        
        return health


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("CF-AI-SDE Trading System")
    print("=" * 60)
    
    try:
        api = TradingSystemAPI()
        
        print("\nRunning health check...")
        health = api.health_check()
        for component, is_healthy in health.items():
            status = "✓" if is_healthy else "✗"
            print(f"  {status} {component}")
        
        print("\nSystem ready for use!")
        print("\nExample usage:")
        print("  api = TradingSystemAPI()")
        print("  results = api.run_full_pipeline(")
        print("      symbols=['AAPL'],")
        print("      start_date='2023-01-01',")
        print("      end_date='2023-12-31',")
        print("      strategy_name='rsi'")
        print("  )")
        
    except Exception as e:
        print(f"\nError initializing system: {e}")
        print("\nPlease ensure:")
        print("  1. config.yaml exists in the root directory")
        print("  2. All required modules are installed (pip install -r requirements.txt)")
        print("  3. Environment variables are set (.env file)")
