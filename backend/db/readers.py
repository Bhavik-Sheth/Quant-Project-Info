"""
MongoDB Readers Module
Handles reading data from MongoDB collections.
Provides fallback behavior when MongoDB is unavailable.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from .connection import get_collection, is_mongodb_available

logger = logging.getLogger(__name__)


class MarketDataReader:
    """Reader for market data collections."""

    @staticmethod
    def read_raw(
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read raw OHLCV data from market_data_raw collection.
        Returns None if MongoDB unavailable or no data found.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for raw data read")
            return None

        collection = get_collection("market_data_raw")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort("timestamp", DESCENDING)

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No raw data found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            # Clean up MongoDB-specific fields
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            # Set timestamp as index if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} raw records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read raw data: {e}")
            return None

    @staticmethod
    def read_clean(
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Read clean OHLCV data from market_data_clean collection."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for clean data read")
            return None

        collection = get_collection("market_data_clean")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort("timestamp", DESCENDING)

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No clean data found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} clean records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read clean data: {e}")
            return None

    @staticmethod
    def get_latest(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the latest raw data record for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("market_data_raw")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"symbol": symbol, "timeframe": timeframe},
                sort=[("timestamp", DESCENDING)],
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get latest data: {e}")
            return None


class FeatureReader:
    """Reader for feature data and normalization parameters."""

    @staticmethod
    def read_features(
        symbol: str,
        timeframe: str,
        version: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read feature data from market_features collection.
        If version not specified, reads latest version.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for features read")
            return None

        collection = get_collection("market_features")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}

            if version:
                query["version"] = version

            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date

            cursor = collection.find(query).sort(
                [("version", DESCENDING), ("timestamp", DESCENDING)]
            )

            if limit:
                cursor = cursor.limit(limit)

            records = list(cursor)

            if not records:
                logger.debug(f"No features found for {symbol}/{timeframe}")
                return None

            df = pd.DataFrame(records)

            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            logger.info(f"Read {len(df)} feature records for {symbol}/{timeframe}")
            return df

        except PyMongoError as e:
            logger.error(f"Failed to read features: {e}")
            return None

    @staticmethod
    def get_normalization_params(
        symbol: str, timeframe: str, version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get normalization parameters for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("normalization_params")
        if collection is None:
            return None

        try:
            query = {"symbol": symbol, "timeframe": timeframe}
            if version:
                query["version"] = version

            record = collection.find_one(query, sort=[("version", DESCENDING)])

            if record:
                if "_id" in record:
                    del record["_id"]
                return record.get("parameters", record)

            return None

        except PyMongoError as e:
            logger.error(f"Failed to get normalization params: {e}")
            return None

    @staticmethod
    def get_returns_and_indicators(
        symbol: str, timeframe: str, lookback_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get returns and key indicators for agent context assembly.
        Returns dict with returns, volatility metrics, and momentum indicators.
        """
        if not is_mongodb_available():
            return None

        collection = get_collection("market_features")
        if collection is None:
            return None

        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)

            records = list(
                collection.find(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": {"$gte": start_date, "$lte": end_date},
                    },
                    sort=[("timestamp", DESCENDING)],
                ).limit(lookback_days * 2)
            )  # Extra buffer for weekends/holidays

            if not records:
                return None

            # Extract key metrics for agent context
            latest = records[0]

            context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "latest_timestamp": latest.get("timestamp"),
                "returns": {},
                "volatility": {},
                "momentum": {},
                "trend": {},
            }

            # Returns
            for key in ["Returns", "Log_Returns", "returns", "log_returns"]:
                if key in latest:
                    context["returns"][key.lower()] = latest[key]

            # Volatility
            for key in ["ATR", "ATR_14", "Volatility", "BB_Width", "atr", "volatility"]:
                if key in latest:
                    context["volatility"][key.lower()] = latest[key]

            # Momentum
            for key in [
                "RSI",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "Stoch_K",
                "Stoch_D",
                "rsi",
                "macd",
                "stoch_k",
                "stoch_d",
            ]:
                if key in latest:
                    context["momentum"][key.lower()] = latest[key]

            # Trend
            for key in [
                "SMA_20",
                "SMA_50",
                "EMA_12",
                "EMA_26",
                "ADX",
                "Trend",
                "sma_20",
                "sma_50",
                "ema_12",
                "ema_26",
                "adx",
            ]:
                if key in latest:
                    context["trend"][key.lower()] = latest[key]

            return context

        except PyMongoError as e:
            logger.error(f"Failed to get returns and indicators: {e}")
            return None


class AgentOutputReader:
    """Reader for agent outputs."""

    @staticmethod
    def get_latest_output(agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest output for a specific agent."""
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for agent output read")
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"agent_name": agent_name}, sort=[("created_at", DESCENDING)]
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get latest agent output: {e}")
            return None

    @staticmethod
    def get_latest_run_outputs(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all agent outputs from the latest run.
        If run_id not specified, gets outputs from most recent run.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning empty list for run outputs")
            return []

        collection = get_collection("agent_outputs")
        if collection is None:
            return []

        try:
            if run_id is None:
                # Get the latest run_id
                latest = collection.find_one(sort=[("created_at", DESCENDING)])
                if latest is None:
                    return []
                run_id = latest.get("run_id")

            records = list(collection.find({"run_id": run_id}))

            for record in records:
                if "_id" in record:
                    del record["_id"]

            logger.info(f"Read {len(records)} agent outputs for run: {run_id}")
            return records

        except PyMongoError as e:
            logger.error(f"Failed to get run outputs: {e}")
            return []

    @staticmethod
    def get_risk_regime_output() -> Optional[Dict[str, Any]]:
        """
        Get the latest risk regime output for UI consumption.
        Returns data formatted for /api/risk-regime endpoint.
        """
        if not is_mongodb_available():
            logger.debug("MongoDB not available, returning None for risk regime")
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            # Try RegimeDetectionAgent first
            record = collection.find_one(
                {"agent_name": "RegimeDetectionAgent"},
                sort=[("created_at", DESCENDING)],
            )

            if record is None:
                # Fallback to RiskMonitoringAgent
                record = collection.find_one(
                    {"agent_name": "RiskMonitoringAgent"},
                    sort=[("created_at", DESCENDING)],
                )

            if record is None:
                return None

            if "_id" in record:
                del record["_id"]

            # Extract response data
            response = record.get("response", {})

            return {
                "agent_name": record.get("agent_name"),
                "timestamp": record.get("timestamp"),
                "created_at": record.get("created_at"),
                "run_id": record.get("run_id"),
                "signal": response.get("signal"),
                "confidence": response.get("confidence"),
                "reasoning": response.get("reasoning"),
                "data": response.get("data", {}),
                "metadata": response.get("metadata", {}),
            }

        except PyMongoError as e:
            logger.error(f"Failed to get risk regime output: {e}")
            return None

    @staticmethod
    def get_aggregated_signals() -> Optional[Dict[str, Any]]:
        """Get the latest aggregated signals from SignalAggregatorAgent."""
        if not is_mongodb_available():
            return None

        collection = get_collection("agent_outputs")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"agent_name": "SignalAggregatorAgent"},
                sort=[("created_at", DESCENDING)],
            )

            if record:
                if "_id" in record:
                    del record["_id"]
                return record

            return None

        except PyMongoError as e:
            logger.error(f"Failed to get aggregated signals: {e}")
            return None


class ValidationLogReader:
    """Reader for validation logs."""

    @staticmethod
    def get_latest_log(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the latest validation log for a symbol/timeframe."""
        if not is_mongodb_available():
            return None

        collection = get_collection("validation_log")
        if collection is None:
            return None

        try:
            record = collection.find_one(
                {"symbol": symbol, "timeframe": timeframe},
                sort=[("validated_at", DESCENDING)],
            )

            if record and "_id" in record:
                del record["_id"]

            return record

        except PyMongoError as e:
            logger.error(f"Failed to get validation log: {e}")
            return None

    @staticmethod
    def get_validation_stats(
        symbol: str, timeframe: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent validation statistics for a symbol/timeframe."""
        if not is_mongodb_available():
            return []

        collection = get_collection("validation_log")
        if collection is None:
            return []

        try:
            records = list(
                collection.find(
                    {"symbol": symbol, "timeframe": timeframe},
                    sort=[("validated_at", DESCENDING)],
                ).limit(limit)
            )

            for record in records:
                if "_id" in record:
                    del record["_id"]

            return records

        except PyMongoError as e:
            logger.error(f"Failed to get validation stats: {e}")
            return []


class MLModelReader:
    """Reader for ML models from MongoDB."""

    def __init__(self, db_client):
        """
        Initialize MLModelReader with database client.
        
        Args:
            db_client: MongoDB database client
        """
        self.db = db_client
        self.collection = self.db['ml_models']

    def load_model(self, model_type: str, version: str = 'latest'):
        """
        Load ML model from MongoDB.
        
        Args:
            model_type: 'direction', 'volatility', 'regime', 'gan'
            version: Specific version or 'latest'
        
        Returns:
            Tuple of (model_object, metadata)
        """
        import pickle
        
        try:
            query = {'model_type': model_type, 'status': 'active'}
            
            if version != 'latest':
                query['version'] = version
            
            # Get latest by trained_at
            document = self.collection.find_one(
                query,
                sort=[('trained_at', DESCENDING)]
            )
            
            if not document:
                raise ValueError(f"No model found for type={model_type}, version={version}")
            
            # Deserialize model
            model_data = document['model_data']
            model = pickle.loads(model_data)
            
            metadata = {
                'model_id': str(document['_id']),
                'version': document['version'],
                'trained_at': document['trained_at'],
                'metrics': document['metrics'],
                'hyperparameters': document.get('hyperparameters', {})
            }
            
            logger.info(f"Loaded {model_type} model (version {metadata['version']})")
            return model, metadata
            
        except PyMongoError as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def list_models(self, model_type: str = None) -> List[Dict]:
        """
        List all available models.
        
        Args:
            model_type: Optional filter by model type
        
        Returns:
            List of model metadata dicts
        """
        try:
            query = {'status': 'active'}
            if model_type:
                query['model_type'] = model_type
            
            models = self.collection.find(query).sort('trained_at', DESCENDING)
            
            result = []
            for m in models:
                result.append({
                    'model_id': str(m['_id']),
                    'model_type': m['model_type'],
                    'version': m['version'],
                    'trained_at': m['trained_at'],
                    'metrics': m['metrics'],
                    'size_mb': m['model_size_mb']
                })
            
            return result
            
        except PyMongoError as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_by_id(self, model_id: str):
        """
        Get specific model by ID.
        
        Args:
            model_id: Model ObjectId as string
        
        Returns:
            Tuple of (model_object, metadata)
        """
        import pickle
        from bson import ObjectId
        
        try:
            document = self.collection.find_one({'_id': ObjectId(model_id)})
            
            if not document:
                raise ValueError(f"Model not found: {model_id}")
            
            model_data = document['model_data']
            model = pickle.loads(model_data)
            
            metadata = {
                'model_id': str(document['_id']),
                'version': document['version'],
                'trained_at': document['trained_at'],
                'metrics': document['metrics'],
                'hyperparameters': document.get('hyperparameters', {})
            }
            
            return model, metadata
            
        except PyMongoError as e:
            logger.error(f"Failed to get model by ID: {e}")
            raise


class AgentMemoryReader:
    """Reader for agent performance history and weights."""

    def __init__(self, db_client):
        """
        Initialize AgentMemoryReader with database client.
        
        Args:
            db_client: MongoDB database client
        """
        self.db = db_client
        self.collection = self.db['agent_memory']

    def get_latest_weight(self, agent_name: str) -> float:
        """
        Get most recent performance weight for an agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Performance weight (default 1.0 if not found)
        """
        try:
            document = self.collection.find_one(
                {'agent_name': agent_name},
                sort=[('timestamp', DESCENDING)]
            )
            
            if document and 'performance_weight' in document:
                logger.info(f"Retrieved weight for {agent_name}: {document['performance_weight']:.3f}")
                return document['performance_weight']
            else:
                logger.info(f"No saved weight for {agent_name}, using default 1.0")
                return 1.0
                
        except PyMongoError as e:
            logger.error(f"Failed to get agent weight: {e}")
            return 1.0

    def get_agent_history(self, agent_name: str, days: int = 30) -> List[Dict]:
        """
        Get agent performance history.
        
        Args:
            agent_name: Name of the agent
            days: Number of days to look back
        
        Returns:
            List of performance records
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            
            history = list(self.collection.find({
                'agent_name': agent_name,
                'timestamp': {'$gte': cutoff}
            }).sort('timestamp', 1))
            
            # Remove _id fields
            for record in history:
                if '_id' in record:
                    del record['_id']
            
            return history
            
        except PyMongoError as e:
            logger.error(f"Failed to get agent history: {e}")
            return []

    def get_agent_accuracy(self, agent_name: str, days: int = 30) -> Optional[float]:
        """
        Calculate agent accuracy over specified period.
        
        Args:
            agent_name: Name of the agent
            days: Number of days to look back
        
        Returns:
            Accuracy as float (0-1) or None
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            
            predictions = list(self.collection.find({
                'agent_name': agent_name,
                'timestamp': {'$gte': cutoff},
                'is_correct': {'$exists': True}
            }))
            
            if not predictions:
                return None
            
            correct = sum(1 for p in predictions if p.get('is_correct', False))
            total = len(predictions)
            
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"{agent_name} accuracy: {accuracy:.2%} ({correct}/{total})")
            return accuracy
            
        except PyMongoError as e:
            logger.error(f"Failed to calculate agent accuracy: {e}")
            return None

    def get_all_agent_weights(self) -> Dict[str, float]:
        """
        Get latest performance weights for all agents.
        
        Returns:
            Dict mapping agent names to performance weights
        """
        try:
            # Aggregate to get latest weight per agent
            pipeline = [
                {'$sort': {'timestamp': -1}},
                {'$group': {
                    '_id': '$agent_name',
                    'performance_weight': {'$first': '$performance_weight'},
                    'timestamp': {'$first': '$timestamp'}
                }}
            ]
            
            results = self.collection.aggregate(pipeline)
            
            weights = {}
            for result in results:
                agent_name = result['_id']
                weight = result['performance_weight']
                weights[agent_name] = weight
            
            logger.info(f"Retrieved weights for {len(weights)} agents")
            return weights
            
        except PyMongoError as e:
            logger.error(f"Failed to get all agent weights: {e}")
            return {}

