"""
Trading Mentor Example Usage

Demonstrates how to use the RAG-based Trading Mentor system.
"""

import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from RAG_Mentor.knowledge.knowledge_loader import KnowledgeLoader
from RAG_Mentor.interface.trading_mentor import TradingMentor
from RAG_Mentor.vector_db.chroma_manager import ChromaManager


def initialize_knowledge_base():
    """Step 1: Initialize the knowledge base with trading principles and news"""
    print("\n" + "="*70)
    print("STEP 1: Initializing Knowledge Base")
    print("="*70)
    
    loader = KnowledgeLoader()
    
    # Load trading principles and sample news
    results = loader.initialize_knowledge_base(include_sample_news=True)
    
    print(f"\n✅ Loaded {results['principles']} trading principles")
    print(f"✅ Loaded {results['news']} news articles")
    
    # Show stats
    stats = loader.get_knowledge_stats()
    print(f"\nKnowledge Base Statistics:")
    for collection, count in stats.items():
        print(f"  - {collection}: {count} documents")
    
    return loader


def create_sample_backtest_data():
    """Create sample backtest data for demonstration"""
    # Sample performance metrics
    performance = {
        'total_return': -0.08,  # -8% (poor performance)
        'sharpe_ratio': 0.45,   # Low risk-adjusted returns
        'max_drawdown': 0.23,   # 23% drawdown (too high)
        'win_rate': 0.42,       # Only 42% win rate
        'profit_factor': 0.95,  # Losing money
        'total_trades': 87,
        'avg_win': 0.035,       # 3.5% average win
        'avg_loss': -0.048      # -4.8% average loss
    }
    
    # Sample trades demonstrating violations
    trades = []
    base_date = datetime(2022, 6, 1)
    
    # Create some trades with violations
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for i in range(50):
        symbol = random.choice(symbols)
        action = 'BUY' if i % 2 == 0 else 'SELL'
        price = 100 + random.uniform(-20, 20)
        quantity = random.randint(10, 100)
        
        # Create some losing trades with no stops (>10% loss)
        if i % 8 == 0:  # Occasional big loss
            pnl = -price * quantity * 0.15  # 15% loss
        else:
            pnl = price * quantity * random.uniform(-0.05, 0.08)
        
        trades.append({
            'timestamp': base_date + timedelta(days=i*2, hours=random.randint(9, 16)),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'reason': 'Technical breakout' if action == 'BUY' else 'Take profit'
        })
    
    # Add some averaging down examples
    # Buy AAPL at $150
    trades.append({
        'timestamp': datetime(2022, 9, 10, 10, 0),
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 50,
        'price': 150.0,
        'pnl': 0,
        'reason': 'Initial entry'
    })
    # Buy more AAPL at $135 (averaging down)
    trades.append({
        'timestamp': datetime(2022, 9, 15, 14, 0),
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 135.0,
        'pnl': -750,  # Loss on first position
        'reason': 'Averaging down'
    })
    
    # Regime breakdown showing poor performance in ranging markets
    regime_breakdown = {
        'TRENDING_UP': {'return': 0.12, 'win_rate': 0.65, 'trades': 25},
        'TRENDING_DOWN': {'return': -0.05, 'win_rate': 0.48, 'trades': 15},
        'RANGING': {'return': -0.15, 'win_rate': 0.35, 'trades': 30},
        'VOLATILE': {'return': -0.08, 'win_rate': 0.40, 'trades': 17}
    }
    
    return performance, trades, regime_breakdown


def demonstrate_trading_mentor():
    """Step 2: Demonstrate the Trading Mentor analysis"""
    print("\n" + "="*70)
    print("STEP 2: Running Trading Mentor Analysis")
    print("="*70)
    
    # Initialize Trading Mentor
    mentor = TradingMentor()
    
    # Get sample backtest data
    performance, trades, regime_breakdown = create_sample_backtest_data()
    
    print(f"\n📊 Analyzing backtest with:")
    print(f"   - Total Return: {performance['total_return']:.2%}")
    print(f"   - Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   - Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"   - {len(trades)} trades")
    
    # Run comprehensive analysis
    print("\n🔍 Generating comprehensive analysis (this may take 30-60 seconds)...\n")
    
    results = mentor.analyze_backtest(
        performance_summary=performance,
        trades=trades,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        regime_breakdown=regime_breakdown
    )
    
    # Display results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print("\n" + "-"*70)
    print("1. EXECUTIVE SUMMARY")
    print("-"*70)
    print(results['summary'])
    
    print("\n" + "-"*70)
    print("2. PERFORMANCE ANALYSIS")
    print("-"*70)
    print(results['performance_analysis'][:1000] + "...")  # First 1000 chars
    
    print("\n" + "-"*70)
    print("3. PRINCIPLE VIOLATIONS")
    print("-"*70)
    print(results['violation_report'][:1000] + "...")  # First 1000 chars
    
    print("\n" + "-"*70)
    print("4. IMPROVEMENT SUGGESTIONS")
    print("-"*70)
    print(results['improvement_suggestions'][:1000] + "...")  # First 1000 chars
    
    return mentor


def demonstrate_conversational_interface(mentor):
    """Step 3: Demonstrate conversational Q&A"""
    print("\n" + "="*70)
    print("STEP 3: Conversational Interface Demo")
    print("="*70)
    
    queries = [
        "Why did the strategy fail during March 2020?",
        "What should I do about the low win rate?",
        "Compare this strategy to buy-and-hold"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n💬 Question {i}: {query}")
        print("-" * 70)
        
        response = mentor.ask_question(query)
        print(response[:500] + "..." if len(response) > 500 else response)
        print()


def save_full_report(mentor):
    """Save complete analysis to file"""
    print("\n" + "="*70)
    print("STEP 4: Saving Complete Report")
    print("="*70)
    
    # Save conversation history
    report_path = "RAG_Mentor/trading_mentor_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("TRADING MENTOR ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        history = mentor.get_conversation_history()
        for msg in history:
            f.write(f"\n[{msg['role'].upper()}]\n")
            f.write(msg['content'])
            f.write("\n" + "-" * 70 + "\n")
    
    print(f"✅ Full report saved to: {report_path}")


def main():
    """Main execution"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                   RAG TRADING MENTOR DEMONSTRATION                    ║
║                                                                       ║
║  This demo showcases the RAG-based Trading Mentor system with:       ║
║  - ChromaDB vector database                                          ║
║  - Gemini/Groq LLM integration                                       ║
║  - Trading principles knowledge base                                 ║
║  - Performance analysis & principle checking                         ║
║  - Conversational Q&A interface                                      ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 1: Initialize knowledge base
        loader = initialize_knowledge_base()
        
        # Step 2: Run Trading Mentor analysis
        mentor = demonstrate_trading_mentor()
        
        # Step 3: Conversational interface
        demonstrate_conversational_interface(mentor)
        
        # Step 4: Save report
        save_full_report(mentor)
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETE!")
        print("="*70)
        print("\nThe Trading Mentor is now ready for use!")
        print("\nQuick Start:")
        print("  from RAG_Mentor.interface.trading_mentor import TradingMentor")
        print("  mentor = TradingMentor()")
        print("  results = mentor.analyze_backtest(performance, trades)")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
