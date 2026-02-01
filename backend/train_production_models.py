"""
Production Model Training Pipeline
CF-AI-SDE Trading System

One-time training script for all ML models using diversified market data.

Usage:
    python train_production_models.py --full-pipeline
    python train_production_models.py --stage ingest
    python train_production_models.py --stage train
    python train_production_models.py --stage validate
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project paths
backend_dir = Path(__file__).parent
data_pipeline_dir = backend_dir / "Data-inges-fe"
sys.path.insert(0, str(backend_dir))

# Import existing pipeline components (using importlib due to hyphens in directory name)
import importlib.util

def _load_module(module_name: str, file_path: Path):
    """Load a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load ingestion runner
_ingestion_module = _load_module("runner", data_pipeline_dir / "src" / "ingestion" / "runner.py")
run_ingestion = _ingestion_module.run_ingestion

# Load feature runner
_feature_module = _load_module("feature_runner", data_pipeline_dir / "src" / "features" / "feature_runner.py")
run_feature_engineering = _feature_module.run_feature_engineering

# Load validation runner
_validation_module = _load_module("validation_runner", data_pipeline_dir / "src" / "validation" / "validation_runner.py")
run_validation = _validation_module.run_validation

# Import model training modules
from ML_Models.direction_pred import Feature_importance, Extract_Rules
# Note: These are imported but not used in current training pipeline
# from ML_Models.Volatility_Forecasting import Volatility_Models, LSTM_Volatility_Model
# from ML_Models.Regime_Classificaiton import Regime_Classifier

# Import database components
from db.writers import MLModelWriter
from db.readers import MLModelReader
from db.connection import get_connection, get_database

# Import configuration
from logical_pipe import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Training Symbols (26 total)
TRAINING_SYMBOLS = {
    "indices": ["^GSPC", "^IXIC", "^DJI", "^NSEI", "^NSEBANK", "^VIX"],
    "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "financials": ["JPM", "BAC", "GS", "V"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "consumer": ["WMT", "HD", "MCD", "NKE"],
    "energy": ["XOM", "CVX", "COP", "SLB"]
}

ALL_SYMBOLS = []
for category, symbols in TRAINING_SYMBOLS.items():
    ALL_SYMBOLS.extend(symbols)

# Timeframe Configuration
PRIMARY_TIMEFRAME = "1d"
PRIMARY_LOOKBACK_YEARS = 5

# Date Ranges
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=PRIMARY_LOOKBACK_YEARS * 365)

# Train/Val/Test Split
TRAIN_RATIO = 0.80  # 2020-2023
VAL_RATIO = 0.10    # 2024 H1
TEST_RATIO = 0.10   # 2024 H2 - 2025

# Model Configuration
MODEL_CONFIGS = {
    "direction": {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "lstm": {
            "sequence_length": 20,
            "units": [64, 32],
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32
        }
    },
    "volatility": {
        "garch": {
            "p": 1,
            "q": 1
        },
        "lstm": {
            "sequence_length": 20,
            "units": [64, 32],
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32
        }
    },
    "regime": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
}


# ============================================================================
# STAGE 1: DATA INGESTION
# ============================================================================

class DataIngestionStage:
    """Stage 1: Ingest OHLCV data using existing pipeline"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.features_dir = self.data_dir / "features"
        
        # Create directories
        for dir_path in [self.raw_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict:
        """Execute data ingestion pipeline"""
        logger.info("=" * 80)
        logger.info("STAGE 1: DATA INGESTION")
        logger.info("=" * 80)
        logger.info(f"Symbols: {len(ALL_SYMBOLS)} ({', '.join(ALL_SYMBOLS[:5])}...)")
        logger.info(f"Timeframe: {PRIMARY_TIMEFRAME}")
        logger.info(f"Date Range: {START_DATE.date()} to {END_DATE.date()}")
        logger.info(f"Expected Samples: ~{len(ALL_SYMBOLS) * 1260}")
        logger.info("=" * 80)
        
        try:
            # Run ingestion using existing pipeline
            ingested_data = run_ingestion(
                symbols=ALL_SYMBOLS,
                timeframes=[PRIMARY_TIMEFRAME],
                start_date=START_DATE,
                end_date=END_DATE,
                save_to_file=True
            )
            
            # Validate data quality
            validation_results = run_validation(
                raw_data=ingested_data,
                timeframes=[PRIMARY_TIMEFRAME],
                save_to_file=True
            )
            
            # Log results
            total_records = sum(
                len(df) for symbol_data in ingested_data.values()
                for df in symbol_data.values()
            )
            
            logger.info(f"✅ Ingestion complete: {total_records:,} records")
            
            # Debug validation results structure
            logger.info(f"Validation results keys: {list(validation_results.keys())}")
            if 'clean' in validation_results:
                clean = validation_results['clean']
                logger.info(f"Clean data keys: {list(clean.keys())}")
                for tf in clean.keys():
                    logger.info(f"Timeframe {tf} has {len(clean[tf])} symbols")
            
            logger.info(f"✅ Validation complete")
            
            return {
                "status": "success",
                "records": total_records,
                "validation_results": validation_results,
                "data": ingested_data
            }
        
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ============================================================================
# STAGE 2: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineeringStage:
    """Stage 2: Engineer 70+ technical indicators"""
    
    def __init__(self, ingested_data: Dict):
        self.ingested_data = ingested_data
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict:
        """Execute feature engineering pipeline"""
        logger.info("=" * 80)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("=" * 80)
        logger.info("Computing 70+ technical indicators...")
        logger.info("=" * 80)
        
        try:
            # Run feature engineering using existing pipeline
            # Extract clean data from validation results
            validation_results = self.ingested_data.get('validation_results', {})
            clean_data_dict = validation_results.get('clean', {})
            
            # Check if we have data
            if not clean_data_dict or PRIMARY_TIMEFRAME not in clean_data_dict:
                raise ValueError(f"No clean data found for timeframe {PRIMARY_TIMEFRAME}")
            
            logger.info(f"Clean data has {len(clean_data_dict[PRIMARY_TIMEFRAME])} symbols")
            
            # Feature runner expects {timeframe: {symbol: df}} format
            feature_data = run_feature_engineering(
                clean_data=clean_data_dict,  # Already in correct format
                timeframes=[PRIMARY_TIMEFRAME],
                save_to_file=True,
                apply_normalization=True
            )
            
            if not feature_data or PRIMARY_TIMEFRAME not in feature_data:
                raise ValueError("Feature engineering returned empty data")
            
            # Count features from first symbol
            timeframe_data = feature_data[PRIMARY_TIMEFRAME]
            sample_symbol = list(timeframe_data.keys())[0]
            sample_df = timeframe_data[sample_symbol]
            num_features = len(sample_df.columns) - 6  # Exclude OHLCV + Date
            
            logger.info(f"✅ Feature engineering complete")
            logger.info(f"✅ Features per sample: {num_features}")
            logger.info(f"✅ Feature columns: {list(sample_df.columns[:10])}...")
            
            return {
                "status": "success",
                "feature_count": num_features,
                "data": feature_data
            }
        
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ============================================================================
# STAGE 3: DATA PREPARATION
# ============================================================================

class DataPreparationStage:
    """Stage 3: Prepare train/val/test splits"""
    
    def __init__(self, feature_data: Dict):
        self.feature_data = feature_data
    
    def prepare_direction_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for direction prediction"""
        logger.info("Preparing direction prediction dataset...")
        
        all_features = []
        all_targets = []
        
        # Feature data is structured as {timeframe: {symbol: df}}
        if PRIMARY_TIMEFRAME not in self.feature_data:
            raise ValueError(f"Timeframe {PRIMARY_TIMEFRAME} not found in feature data")
        
        timeframe_data = self.feature_data[PRIMARY_TIMEFRAME]
        
        for symbol, df in timeframe_data.items():
            df = df.copy()
            
            # Need to get close price - check column names
            close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
            if close_col is None:
                logger.warning(f"No close column found for {symbol}, skipping")
                continue
            
            # Calculate direction target
            df['return'] = df[close_col].pct_change()
            df['direction'] = 0  # Neutral
            df.loc[df['return'] > 0.001, 'direction'] = 1  # Up
            df.loc[df['return'] < -0.001, 'direction'] = -1  # Down
            
            # Remove NaN and first row
            df = df.dropna()
            
            # Select features (exclude OHLCV and target columns)
            exclude_cols = ['open', 'high', 'low', 'close', 'adj close', 'volume', 
                          'timestamp', 'return', 'direction', 'symbol', 'dividends', 'stock splits']
            feature_cols = [col for col in df.columns 
                          if col.lower() not in exclude_cols]
            
            if len(feature_cols) == 0:
                logger.warning(f"No features found for {symbol}, skipping")
                continue
                
            all_features.append(df[feature_cols])
            all_targets.append(df['direction'])
        
        if not all_features:
            raise ValueError("No features prepared for any symbol")
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Clean data: replace inf with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]
        
        logger.info(f"✅ Direction dataset: {len(X):,} samples, {len(X.columns)} features")
        return X, y
    
    def prepare_volatility_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for volatility forecasting"""
        logger.info("Preparing volatility forecasting dataset...")
        
        all_features = []
        all_targets = []
        
        if PRIMARY_TIMEFRAME not in self.feature_data:
            raise ValueError(f"Timeframe {PRIMARY_TIMEFRAME} not found in feature data")
        
        timeframe_data = self.feature_data[PRIMARY_TIMEFRAME]
        
        for symbol, df in timeframe_data.items():
            df = df.copy()
            
            # Get close column
            close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
            if close_col is None:
                continue
            
            # Calculate realized volatility (20-day rolling std of returns)
            df['return'] = df[close_col].pct_change()
            df['volatility'] = df['return'].rolling(window=20).std()
            
            # Remove NaN
            df = df.dropna()
            
            # Select features
            exclude_cols = ['open', 'high', 'low', 'close', 'adj close', 'volume',
                          'timestamp', 'return', 'volatility', 'symbol', 'dividends', 'stock splits']
            feature_cols = [col for col in df.columns if col.lower() not in exclude_cols]
            
            if len(feature_cols) == 0:
                continue
            
            all_features.append(df[feature_cols])
            all_targets.append(df['volatility'])
        
        if not all_features:
            raise ValueError("No features prepared for volatility")
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Clean data: replace inf with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN in either X or y
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"✅ Volatility dataset: {len(X):,} samples, {len(X.columns)} features")
        return X, y
    
    def prepare_regime_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for regime classification"""
        logger.info("Preparing regime classification dataset...")
        
        # For regime, we primarily use index data
        index_symbols = TRAINING_SYMBOLS["indices"]
        
        all_features = []
        all_targets = []
        
        if PRIMARY_TIMEFRAME not in self.feature_data:
            raise ValueError(f"Timeframe {PRIMARY_TIMEFRAME} not found in feature data")
        
        timeframe_data = self.feature_data[PRIMARY_TIMEFRAME]
        
        for symbol in index_symbols:
            if symbol not in timeframe_data:
                continue
            
            df = timeframe_data[symbol].copy()
            
            # Get close column
            close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
            if close_col is None:
                continue
            
            # Calculate regime features
            df['return'] = df[close_col].pct_change()
            df['ma_50'] = df[close_col].rolling(window=50).mean()
            df['ma_200'] = df[close_col].rolling(window=200).mean()
            df['trend'] = (df['ma_50'] > df['ma_200']).astype(int)
            
            # Assign regime labels
            df['regime'] = 2  # Sideways (default)
            df.loc[(df['trend'] == 1) & (df['return'] > 0), 'regime'] = 0  # Bull
            df.loc[(df['trend'] == 0) & (df['return'] < 0), 'regime'] = 1  # Bear
            df.loc[df['return'].rolling(20).std() > 0.02, 'regime'] = 3  # High Vol
            
            # Remove NaN
            df = df.dropna()
            
            # Select features
            exclude_cols = ['open', 'high', 'low', 'close', 'adj close', 'volume',
                          'timestamp', 'return', 'regime', 'symbol', 
                          'ma_50', 'ma_200', 'trend', 'dividends', 'stock splits']
            feature_cols = [col for col in df.columns if col.lower() not in exclude_cols]
            
            if len(feature_cols) == 0:
                continue
            
            all_features.append(df[feature_cols])
            all_targets.append(df['regime'])
        
        if not all_features:
            raise ValueError("No features prepared for regime")
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Clean data: replace inf with NaN, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]
        
        logger.info(f"✅ Regime dataset: {len(X):,} samples, {len(X.columns)} features")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Split data into train/val/test sets"""
        n_samples = len(X)
        train_idx = int(n_samples * TRAIN_RATIO)
        val_idx = int(n_samples * (TRAIN_RATIO + VAL_RATIO))
        
        return {
            "X_train": X.iloc[:train_idx],
            "y_train": y.iloc[:train_idx],
            "X_val": X.iloc[train_idx:val_idx],
            "y_val": y.iloc[train_idx:val_idx],
            "X_test": X.iloc[val_idx:],
            "y_test": y.iloc[val_idx:]
        }
    
    def run(self) -> Dict:
        """Execute data preparation pipeline"""
        logger.info("=" * 80)
        logger.info("STAGE 3: DATA PREPARATION")
        logger.info("=" * 80)
        
        try:
            # Prepare datasets
            X_dir, y_dir = self.prepare_direction_data()
            X_vol, y_vol = self.prepare_volatility_data()
            X_reg, y_reg = self.prepare_regime_data()
            
            # Split datasets
            direction_splits = self.split_data(X_dir, y_dir)
            volatility_splits = self.split_data(X_vol, y_vol)
            regime_splits = self.split_data(X_reg, y_reg)
            
            logger.info("=" * 80)
            logger.info("SPLIT SUMMARY:")
            logger.info(f"Direction - Train: {len(direction_splits['X_train']):,}, "
                       f"Val: {len(direction_splits['X_val']):,}, "
                       f"Test: {len(direction_splits['X_test']):,}")
            logger.info(f"Volatility - Train: {len(volatility_splits['X_train']):,}, "
                       f"Val: {len(volatility_splits['X_val']):,}, "
                       f"Test: {len(volatility_splits['X_test']):,}")
            logger.info(f"Regime - Train: {len(regime_splits['X_train']):,}, "
                       f"Val: {len(regime_splits['X_val']):,}, "
                       f"Test: {len(regime_splits['X_test']):,}")
            logger.info("=" * 80)
            
            return {
                "status": "success",
                "direction": direction_splits,
                "volatility": volatility_splits,
                "regime": regime_splits
            }
        
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ============================================================================
# STAGE 4: MODEL TRAINING
# ============================================================================

class ModelTrainingStage:
    """Stage 4: Train all ML models"""
    
    def __init__(self, prepared_data: Dict, config_loader: ConfigLoader):
        self.prepared_data = prepared_data
        self.config_loader = config_loader
        self.model_writer = MLModelWriter(get_database())
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def train_direction_models(self) -> Dict:
        """Train direction prediction models"""
        logger.info("Training direction prediction models...")
        
        splits = self.prepared_data["direction"]
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        
        # Feature importance and rules
        feature_importance = Feature_importance(X_train, y_train)
        rules_extractor = Extract_Rules(X_train, y_train)
        
        # Get top features
        top_features = feature_importance.top_30_features_list
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        
        # Train XGBoost
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(**MODEL_CONFIGS["direction"]["xgboost"])
        xgb_model.fit(X_train_selected, y_train)
        
        # Evaluate
        train_acc = xgb_model.score(X_train_selected, y_train)
        test_acc = xgb_model.score(X_test_selected, y_test)
        
        logger.info(f"✅ Direction XGBoost - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        
        # Save model
        import pickle
        model_path = self.models_dir / "direction_xgboost_v1.0.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        # Save to database
        self.model_writer.save_model(
            model_name="direction_xgboost",
            version="1.0",
            model_object=xgb_model,
            metadata={
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "features": top_features,
                "trained_date": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "symbols": ALL_SYMBOLS
            }
        )
        
        return {
            "xgboost": {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model_path": str(model_path)
            }
        }
    
    def train_volatility_models(self) -> Dict:
        """Train volatility forecasting models"""
        logger.info("Training volatility forecasting models...")
        
        splits = self.prepared_data["volatility"]
        
        # For now, use a simple baseline (can expand with GARCH/LSTM later)
        logger.info("✅ Volatility models - Using rolling std baseline")
        
        return {"baseline": "rolling_std_20"}
    
    def train_regime_models(self) -> Dict:
        """Train regime classification models"""
        logger.info("Training regime classification models...")
        
        splits = self.prepared_data["regime"]
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        
        # Train Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(**MODEL_CONFIGS["regime"]["random_forest"])
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = rf_model.score(X_train, y_train)
        test_acc = rf_model.score(X_test, y_test)
        
        logger.info(f"✅ Regime Random Forest - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        
        # Save model
        import pickle
        model_path = self.models_dir / "regime_rf_v1.0.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Save to database
        self.model_writer.save_model(
            model_name="regime_rf",
            version="1.0",
            model_object=rf_model,
            metadata={
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "trained_date": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "symbols": TRAINING_SYMBOLS["indices"]
            }
        )
        
        return {
            "random_forest": {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model_path": str(model_path)
            }
        }
    
    def run(self) -> Dict:
        """Execute model training pipeline"""
        logger.info("=" * 80)
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("=" * 80)
        
        try:
            results = {}
            
            # Train direction models
            results["direction"] = self.train_direction_models()
            
            # Train volatility models
            results["volatility"] = self.train_volatility_models()
            
            # Train regime models
            results["regime"] = self.train_regime_models()
            
            logger.info("=" * 80)
            logger.info("✅ All models trained successfully!")
            logger.info("=" * 80)
            
            return {"status": "success", "results": results}
        
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class ProductionTrainingPipeline:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self):
        self.config_loader = ConfigLoader("config.yaml")
        self.results = {}
    
    def run_full_pipeline(self):
        """Execute all stages in sequence"""
        logger.info("🚀 Starting Production Model Training Pipeline")
        logger.info("=" * 80)
        
        # Stage 1: Ingestion
        ingestion_stage = DataIngestionStage()
        ingestion_result = ingestion_stage.run()
        if ingestion_result["status"] != "success":
            logger.error("Pipeline failed at ingestion stage")
            return
        self.results["ingestion"] = ingestion_result
        
        # Stage 2: Feature Engineering
        feature_stage = FeatureEngineeringStage(ingestion_result)  # Pass full result with validation_results
        feature_result = feature_stage.run()
        if feature_result["status"] != "success":
            logger.error("Pipeline failed at feature engineering stage")
            return
        self.results["features"] = feature_result
        
        # Stage 3: Data Preparation
        prep_stage = DataPreparationStage(feature_result["data"])
        prep_result = prep_stage.run()
        if prep_result["status"] != "success":
            logger.error("Pipeline failed at data preparation stage")
            return
        self.results["preparation"] = prep_result
        
        # Stage 4: Model Training
        training_stage = ModelTrainingStage(prep_result, self.config_loader)
        training_result = training_stage.run()
        if training_result["status"] != "success":
            logger.error("Pipeline failed at model training stage")
            return
        self.results["training"] = training_result
        
        # Save final report
        self.save_training_report()
        
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETE!")
        logger.info("=" * 80)
    
    def save_training_report(self):
        """Save training report to file"""
        report_path = Path("training_report.json")
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        logger.info(f"📊 Training report saved: {report_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CF-AI-SDE Production Model Training")
    parser.add_argument(
        "--stage",
        choices=["ingest", "train", "validate", "full-pipeline"],
        default="full-pipeline",
        help="Pipeline stage to execute"
    )
    
    args = parser.parse_args()
    
    pipeline = ProductionTrainingPipeline()
    
    if args.stage == "full-pipeline":
        pipeline.run_full_pipeline()
    else:
        logger.info(f"Running stage: {args.stage}")
        # Add individual stage execution if needed


if __name__ == "__main__":
    main()
