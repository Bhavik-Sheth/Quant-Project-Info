"""
Ensemble Orchestrator Module

Dynamic strategy selection and aggregation using LangChain/LangGraph.
Uses LLMs to decide which strategies to activate based on market regime.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Add parent directory to path
# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from quant_strategy.base import Signal, Action, Context, Regime, BaseStrategy

# LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Using fallback orchestration.")

# LangGraph imports
try:
    from langgraph.graph import Graph, StateGraph
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not installed. Using simple orchestration.")


class ConflictResolver:
    """
    Detects and resolves contradictory signals using LLM reasoning.
    
    Example conflict: RSI=Oversold (BUY) but Regime=Bearish (SELL)
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize conflict resolver.
        
        Args:
            llm_client: LangChain LLM instance (optional)
        """
        self.llm_client = llm_client
    
    def detect_conflicts(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """
        Identify contradictory signals.
        
        Args:
            signals: List of signals from different strategies
            
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # Group by action
        buy_signals = [s for s in signals if s.action == Action.BUY]
        sell_signals = [s for s in signals if s.action == Action.SELL]
        
        # Conflict if both BUY and SELL signals exist
        if buy_signals and sell_signals:
            conflicts.append({
                'type': 'OPPOSING_ACTIONS',
                'buy_strategies': [s.strategy_name for s in buy_signals],
                'sell_strategies': [s.strategy_name for s in sell_signals],
                'description': f"{len(buy_signals)} strategies say BUY, {len(sell_signals)} say SELL"
            })
        
        # Check confidence contradictions
        high_conf_buy = [s for s in buy_signals if s.confidence > 0.8]
        high_conf_sell = [s for s in sell_signals if s.confidence > 0.8]
        
        if high_conf_buy and high_conf_sell:
            conflicts.append({
                'type': 'HIGH_CONFIDENCE_CONFLICT',
                'description': 'Multiple high-confidence signals disagree'
            })
        
        return conflicts
    
    def resolve_with_llm(self, signals: List[Signal], context: Context) -> Dict[str, Any]:
        """
        Use LLM to resolve conflicts and explain decision.
        
        Args:
            signals: All signals
            context: Market context
            
        Returns:
            Resolution with explanation
        """
        if not self.llm_client:
            return self._fallback_resolution(signals)
        
        # Prepare prompt
        signals_text = "\n".join([
            f"- {s.strategy_name}: {s.action.value} (confidence {s.confidence:.0%}) - {s.reason}"
            for s in signals
        ])
        
        prompt = f"""Given these conflicting trading signals for {context.symbol}, resolve the conflict:

{signals_text}

Market Context:
- Regime: {context.current_regime.value}
- Price: ${context.price:.2f}

Provide:
1. Final decision (BUY/SELL/HOLD)
2. Confidence level (0-100%)
3. Clear reasoning explaining which signals to prioritize and why

Format: JSON with keys 'action', 'confidence', 'reasoning'
"""
        
        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            return result
        except Exception as e:
            print(f"LLM resolution failed: {e}")
            return self._fallback_resolution(signals)
    
    def _fallback_resolution(self, signals: List[Signal]) -> Dict[str, Any]:
        """Simple voting fallback if LLM unavailable."""
        if not signals:
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No signals'}
        
        # Weighted voting by confidence
        action_votes = {}
        for signal in signals:
            vote = signal.action.value
            weight = signal.confidence
            action_votes[vote] = action_votes.get(vote, 0.0) + weight
        
        # Winner
        winner = max(action_votes, key=action_votes.get)
        total_weight = sum(action_votes.values())
        confidence = action_votes[winner] / total_weight if total_weight > 0 else 0.0
        
        return {
            'action': winner,
            'confidence': confidence,
            'reasoning': f"Weighted vote: {action_votes}"
        }


class StrategyOrchestrator:
    """
    Dynamic strategy orchestrator using LangChain/LangGraph.
    
    Core Innovation:
    - LLM decides which strategies to activate based on market regime
    - LangGraph executes strategies in parallel
    - LLM synthesizes final decision with natural language reasoning
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        regime_map: Optional[Dict[Regime, List[str]]] = None,
        api_key: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            strategies: List of strategy instances
            regime_map: {Regime: [strategy_names]} mapping
            api_key: Gemini API key (reads from env if None)
            use_llm: Whether to use LLM for orchestration
        """
        self.strategies = {s.name: s for s in strategies}
        self.regime_map = regime_map or self._default_regime_map()
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        
        # Initialize LLM
        if self.use_llm:
            if api_key is None:
                import os
                api_key = os.getenv('GEMINI_API_KEY')
            
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.2  # Lower temp for more consistent decisions
                )
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
        
        self.conflict_resolver = ConflictResolver(self.llm)
        self.decision_cache: Dict[str, Any] = {}  # Cache to reduce API calls
    
    def _default_regime_map(self) -> Dict[Regime, List[str]]:
        """Default regime-to-strategy mapping."""
        return {
            Regime.BULLISH: ['MomentumStrategy', 'TrendFollowing'],
            Regime.BEARISH: ['ShortMomentum', 'DefensiveStrategy'],
            Regime.RANGING: ['MeanReversion', 'BollingerBands', 'RSIStrategy'],
            Regime.VOLATILE: ['VolArbitrage', 'OptionsStrategy'],
            Regime.CRISIS: ['DefensiveStrategy']  # Minimal exposure
        }
    
    def decide(self, context: Context) -> Signal:
        """
        Main orchestration logic.
        
        Workflow:
        1. Detect regime (from context)
        2. Select strategies for regime
        3. Execute strategies in parallel
        4. Collect signals
        5. LLM synthesis or weighted voting
        6. Return final signal
        
        Args:
            context: Market state at time t
            
        Returns:
            Aggregated Signal with full reasoning
        """
        # Step 1: Get active strategies for current regime
        active_strategy_names = self.regime_map.get(context.current_regime, [])
        active_strategies = [
            self.strategies[name] for name in active_strategy_names
            if name in self.strategies
        ]
        
        if not active_strategies:
            # Fallback: Use all strategies
            active_strategies = list(self.strategies.values())
        
        # Step 2: Adjust parameters based on volatility
        if 'predicted_volatility' in context.ml_predictions:
            vol = context.ml_predictions['predicted_volatility']
            for strategy in active_strategies:
                strategy.adjust_parameters(vol)
        
        # Step 3: Generate signals from each strategy
        signals = []
        for strategy in active_strategies:
            try:
                signal = strategy.generate_signal(context)
                signals.append(signal)
            except Exception as e:
                print(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not signals:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No strategies generated valid signals",
                strategy_name="Orchestrator"
            )
        
        # Step 4: Check conflicts
        conflicts = self.conflict_resolver.detect_conflicts(signals)
        
        # Step 5: Synthesize final decision
        if self.use_llm and conflicts:
            # Use LLM for complex conflict resolution
            final_decision = self._llm_synthesis(signals, context, conflicts)
        elif self.use_llm:
            # Use LLM even without conflicts for better reasoning
            final_decision = self._llm_synthesis(signals, context, [])
        else:
            # Fallback to weighted voting
            final_decision = self._weighted_voting(signals, context)
        
        return final_decision
    
    def _llm_synthesis(
        self,
        signals: List[Signal],
        context: Context,
        conflicts: List[Dict]
    ) -> Signal:
        """
        Use LLM to synthesize final decision from multiple signals.
        
        Generates natural language reasoning for explainability.
        """
        # Check cache
        cache_key = f"{context.symbol}_{context.current_regime.value}_{len(signals)}"
        if cache_key in self.decision_cache:
            cached = self.decision_cache[cache_key]
            # Reuse cached logic but update timestamp
            return Signal(
                action=Action[cached['action']],
                confidence=cached['confidence'],
                reason=cached['reasoning'] + " [CACHED]",
                strategy_name="LLM-Orchestrator",
                metadata={'cached': True, 'signals_count': len(signals)}
            )
        
        # Prepare comprehensive prompt
        signals_summary = "\n".join([
            f"{i+1}. {s.strategy_name}: {s.action.value} (confidence {s.confidence:.0%})\n   Reason: {s.reason}"
            for i, s in enumerate(signals)
        ])
        
        ml_info = ""
        if context.ml_predictions:
            ml_info = f"\nML Predictions: {json.dumps(context.ml_predictions, indent=2)}"
        
        conflict_info = ""
        if conflicts:
            conflict_info = f"\nCONFLICTS DETECTED:\n" + "\n".join([
                f"- {c['description']}" for c in conflicts
            ])
        
        prompt = f"""You are a quantitative trading strategist synthesizing signals for {context.symbol}.

MARKET CONTEXT:
- Regime: {context.current_regime.value}
- Price: ${context.price:.2f}
- Timestamp: {context.timestamp}
{ml_info}

STRATEGY SIGNALS:
{signals_summary}
{conflict_info}

TASK:
Synthesize these signals into ONE final decision. Consider:
1. Regime appropriateness (e.g., mean reversion works in ranging markets)
2. ML confidence (boost strategies that align with high-confidence ML predictions)
3. Signal consensus vs. conflicts

OUTPUT (JSON):
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100,
  "reasoning": "Clear explanation of why this decision was made and which signals were prioritized"
}}
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert quantitative trading strategist."),
                HumanMessage(content=prompt)
            ])
            
            # Parse response
            content = response.content
            # Try to extract JSON
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '{' in content:
                json_str = content[content.find('{'):content.rfind('}')+1]
            else:
                json_str = content
            
            result = json.loads(json_str)
            
            # Cache for future
            self.decision_cache[cache_key] = result
            
            return Signal(
                action=Action[result['action'].upper()],
                confidence=float(result['confidence']) / 100.0,
                reason=result['reasoning'],
                strategy_name="LLM-Orchestrator",
                metadata={
                    'llm_synthesis': True,
                    'signals_count': len(signals),
                    'conflicts': len(conflicts)
                }
            )
            
        except Exception as e:
            print(f"LLM synthesis failed: {e}, falling back to weighted voting")
            return self._weighted_voting(signals, context)
    
    def _weighted_voting(self, signals: List[Signal], context: Context) -> Signal:
        """
        Fallback: Weighted voting by confidence.
        
        Optionally boost weights for signals that align with ML predictions.
        """
        if not signals:
            return Signal(
                action=Action.HOLD,
                confidence=0.0,
                reason="No signals to aggregate",
                strategy_name="Orchestrator"
            )
        
        # Calculate weights
        action_scores = {}
        for signal in signals:
            vote = signal.action
            weight = signal.confidence
            
            # ML confidence boost
            if 'direction_confidence' in context.ml_predictions:
                ml_conf = context.ml_predictions['direction_confidence']
                # If ML agrees with signal, boost weight
                if ml_conf > 0.7:
                    weight *= 1.5  # 50% boost
            
            if vote not in action_scores:
                action_scores[vote] = {'total': 0.0, 'reasons': []}
            
            action_scores[vote]['total'] += weight
            action_scores[vote]['reasons'].append(f"{signal.strategy_name}: {signal.reason}")
        
        # Winner
        winner_action = max(action_scores, key=lambda a: action_scores[a]['total'])
        total_weight = sum(scores['total'] for scores in action_scores.values())
        confidence = action_scores[winner_action]['total'] / total_weight if total_weight > 0 else 0.0
        
        # Aggregate reasoning
        reasons = action_scores[winner_action]['reasons']
        aggregated_reason = f"Weighted vote: {winner_action.value} ({confidence:.0%} confidence). "
        aggregated_reason += "Supporting signals: " + "; ".join(reasons[:3])  # Top 3
        
        return Signal(
            action=winner_action,
            confidence=confidence,
            reason=aggregated_reason,
            strategy_name="Orchestrator-WeightedVote",
            metadata={
                'voting_scores': {a.value: s['total'] for a, s in action_scores.items()},
                'total_signals': len(signals)
            }
        )
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a new strategy to the pool."""
        self.strategies[strategy.name] = strategy
    
    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the pool."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
