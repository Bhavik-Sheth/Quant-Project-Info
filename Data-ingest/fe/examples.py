"""
Example usage of the financial data pipeline.
Demonstrates how to use the pipeline for different scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.runner import run_ingestion
from src.validation.validation_runner import run_validation
from src.features.feature_runner import run_feature_engineering
from main import run_full_pipeline, run_stage_independently
import pandas as pd


def example_1_full_pipeline():
    """Example 1: Run complete pipeline with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Full Pipeline Execution")
    print("="*80)
    
    results = run_full_pipeline()
    return results


def example_2_custom_symbols():
    """Example 2: Run pipeline for specific symbols and timeframes."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Symbols and Timeframes")
    print("="*80)
    
    custom_symbols = ["AAPL", "MSFT", "^GSPC"]
    custom_timeframes = ["1h", "1d"]
    
    results = run_full_pipeline(
        symbols=custom_symbols,
        timeframes=custom_timeframes,
        save_to_file=True
    )
    
    return results


def example_3_individual_stages():
    """Example 3: Run individual pipeline stages."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Running Individual Stages")
    print("="*80)
    
    # Run only ingestion
    print("\nRunning ingestion only...")
    run_stage_independently('ingestion', 
                           symbols=["AAPL"], 
                           timeframes=["1d"])
    
    # Run only validation (loads from files)
    print("\nRunning validation only...")
    run_stage_independently('validation')
    
    # Run only feature engineering (loads from files)
    print("\nRunning feature engineering only...")
    run_stage_independently('features')


def example_4_load_and_analyze():
    """Example 4: Load generated features and perform basic analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Loading and Analyzing Generated Features")
    print("="*80)
    
    # Load feature data
    df = pd.read_csv('data/features/1d/AAPL_1d_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\nLoaded {len(df)} records for AAPL (1d)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nAvailable features ({len(df.columns)} total):")
    
    # Categorize features
    base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    adjusted_cols = [col for col in df.columns if 'adj_' in col]
    technical_cols = [col for col in df.columns if col not in base_cols + adjusted_cols 
                     and col not in ['dividends', 'stock splits']]
    
    print(f"  - Base OHLCV: {len(base_cols)}")
    print(f"  - Adjusted prices: {len(adjusted_cols)}")
    print(f"  - Technical indicators: {len(technical_cols)}")
    
    print(f"\nTechnical indicators available:")
    for col in sorted(technical_cols):
        print(f"  - {col}")
    
    # Basic statistics
    print(f"\nRecent price statistics:")
    print(df[['close', 'sma_20', 'rsi_14', 'atr_14']].tail(10))
    
    # Check for NaN values (expected in early periods due to indicator warmup)
    print(f"\nNaN values per column (first 200 periods):")
    nan_counts = df.head(200).isna().sum()
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"  {col}: {count}")
    
    return df


def example_5_multi_symbol_analysis():
    """Example 5: Compare features across multiple symbols."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Multi-Symbol Feature Comparison")
    print("="*80)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dfs = {}
    
    for symbol in symbols:
        filepath = f'data/features/1d/{symbol}_1d_features.csv'
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs[symbol] = df
        
    print(f"\nLoaded {len(symbols)} symbols")
    
    # Compare latest RSI values
    print("\nLatest RSI (14) values:")
    for symbol, df in dfs.items():
        latest_rsi = df['rsi_14'].iloc[-1]
        print(f"  {symbol}: {latest_rsi:.2f}")
    
    # Compare volatility
    print("\nAverage ATR (14) over last 30 days:")
    for symbol, df in dfs.items():
        avg_atr = df['atr_14'].tail(30).mean()
        print(f"  {symbol}: ${avg_atr:.2f}")
    
    return dfs


def example_6_validation_log_analysis():
    """Example 6: Analyze validation issues."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Validation Log Analysis")
    print("="*80)
    
    # Load validation log
    log_df = pd.read_csv('data/validated/validation_log.csv')
    
    print(f"\nTotal validation issues: {len(log_df)}")
    
    # Issues by type
    print("\nIssues by type:")
    issue_counts = log_df['issue'].value_counts()
    for issue, count in issue_counts.items():
        print(f"  {issue}: {count}")
    
    # Issues by symbol
    print("\nIssues by symbol:")
    symbol_counts = log_df['symbol'].value_counts()
    for symbol, count in symbol_counts.head(10).items():
        print(f"  {symbol}: {count}")
    
    return log_df


if __name__ == "__main__":
    # Run examples (comment out the ones you don't want to run)
    
    # Example 1: Full pipeline (already run in main.py)
    # results = example_1_full_pipeline()
    
    # Example 2: Custom configuration
    # results = example_2_custom_symbols()
    
    # Example 3: Individual stages
    # example_3_individual_stages()
    
    # Example 4: Load and analyze features
    df = example_4_load_and_analyze()
    
    # Example 5: Multi-symbol comparison
    dfs = example_5_multi_symbol_analysis()
    
    # Example 6: Validation log analysis
    log_df = example_6_validation_log_analysis()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
