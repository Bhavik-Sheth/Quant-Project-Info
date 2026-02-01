"""
Example Usage: Quantitative Strategy Library

Demonstrates various usage patterns for the quant_strategy library.
"""

import sys
import os
from datetime import datetime, timedelta

# Setup paths
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from quant_strategy import (
    BacktestEngine,
    StrategyOrchestrator,
    RSIStrategy,
    MovingAverageCrossStrategy,
    BollingerBandsStrategy,
    MLSignalFilter,
    VolArbitrageStrategy,
    Regime
)


def example_1_single_strategy():
    """
    Example 1: Run a single RSI strategy standalone.
    
    Simplest usage - one strategy, no orchestration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Single RSI Strategy")
    print("="*60 + "\n")
    
    # Create engine
    engine = BacktestEngine(
        mongo_uri="mongodb://localhost:27017/",
        db_name="trading_db",
        collection_name="market_data",
        initial_capital=100000
    )
    
    # Create simple orchestrator with single strategy
    rsi_strategy = RSIStrategy(oversold=30, overbought=70)
    orchestrator = StrategyOrchestrator(
        strategies=[rsi_strategy],
        use_llm=False  # Disable LLM for simple case
    )
    
    # Run backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        orchestrator=orchestrator,
        symbol="AAPL"
    )
    
    print(f"\nResults: {results}")
    
    # Export trades
    engine.export_trades("results/example1_trades.csv")


def example_2_ml_enhanced():
    """
    Example 2: ML-Enhanced Strategy.
    
    Wraps RSI strategy with ML confidence filter.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: ML-Enhanced RSI Strategy")
    print("="*60 + "\n")
    
    # Create base strategy
    base_rsi = RSIStrategy(name="BaseRSI")
    
    # Wrap with ML filter
    ml_enhanced_rsi = MLSignalFilter(
        base_strategy=base_rsi,
        confidence_threshold=0.70
    )
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator(
        strategies=[ml_enhanced_rsi],
        use_llm=False
    )
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        orchestrator=orchestrator,
        symbol="GOOGL"
    )
    
    print(f"\nML-Enhanced Results: {results}")


def example_3_full_orchestration():
    """
    Example 3: Full Orchestration with LLM.
    
    Multiple strategies, dynamic selection based on regime, LLM synthesis.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Full LLM-Powered Orchestration")
    print("="*60 + "\n")
    
    # Create strategy pool
    rsi_strat = RSIStrategy(name="RSI_MeanRev")
    ma_cross = MovingAverageCrossStrategy(name="MA_Trend")
    bb_strat = BollingerBandsStrategy(name="BB_MeanRev")
    
    # Optional: ML-enhance some strategies
    ml_rsi = MLSignalFilter(rsi_strat, name="ML_RSI")
    
    strategies = [rsi_strat, ma_cross, bb_strat, ml_rsi]
    
    # Define regime mapping
    regime_map = {
        Regime.BULLISH: ["MA_Trend"],
        Regime.BEARISH: ["MA_Trend"],
        Regime.RANGING: ["RSI_MeanRev", "BB_MeanRev", "ML_RSI"],
        Regime.VOLATILE: ["BB_MeanRev"],
        Regime.CRISIS: []  # No trading in crisis
    }
    
    # Create orchestrator with LLM
    orchestrator = StrategyOrchestrator(
        strategies=strategies,
        regime_map=regime_map,
        use_llm=True  # Enable Gemini for synthesis
    )
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        orchestrator=orchestrator,
        symbol="MSFT"
    )
    
    print(f"\nFull Orchestration Results: {results}")
    engine.export_trades("results/example3_orchestrated_trades.csv")


def example_4_options_strategy():
    """
    Example 4: Volatility Arbitrage with Options.
    
    Requires option chain data in MongoDB.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Volatility Arbitrage Strategy")
    print("="*60 + "\n")
    
    # Create vol arb strategy
    vol_arb = VolArbitrageStrategy(
        mispricing_threshold=0.15,  # 15% mispricing required
        option_type="call"
    )
    
    orchestrator = StrategyOrchestrator(
        strategies=[vol_arb],
        use_llm=False
    )
    
    engine = BacktestEngine(initial_capital=50000)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        orchestrator=orchestrator,
        symbol="SPY"  # ETF with liquid options
    )
    
    print(f"\nOptions Strategy Results: {results}")


def main():
    """Run examples based on user choice."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quant Strategy Library Examples")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "ml-enhanced", "orchestration", "options", "all"],
        default="single",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "single" or args.mode == "all":
        try:
            example_1_single_strategy()
        except Exception as e:
            print(f"Example 1 failed: {e}")
    
    if args.mode == "ml-enhanced" or args.mode == "all":
        try:
            example_2_ml_enhanced()
        except Exception as e:
            print(f"Example 2 failed: {e}")
    
    if args.mode == "orchestration" or args.mode == "all":
        try:
            example_3_full_orchestration()
        except Exception as e:
            print(f"Example 3 failed: {e}")
    
    if args.mode == "options":
        try:
            example_4_options_strategy()
        except Exception as e:
            print(f"Example 4 failed: {e}")


if __name__ == "__main__":
    main()
