"""
Utility Module for Mathematical and Financial Calculations

Provides:
- Black-Scholes option pricing
- Kelly Criterion for position sizing
- Volatility-based position sizing
- Risk metrics (Sharpe, Drawdown)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, List, Optional
import math


# ==================== OPTIONS PRICING ====================

def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Black-Scholes option pricing model for European options.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: "call" or "put"
        
    Returns:
        Fair value of the option
        
    Example:
        >>> # Price a call option
        >>> fair_value = black_scholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
        >>> print(f"Call option fair value: ${fair_value:.2f}")
    """
    if T <= 0:
        # At expiration
        if option_type.lower() == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iterations: int = 100,
    tolerance: float = 1e-5
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        option_price: Observed market price of option
        S, K, T, r: Black-Scholes parameters
        option_type: "call" or "put"
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatility (annualized)
    """
    # Initial guess
    sigma = 0.2
    
    for _ in range(max_iterations):
        price = black_scholes(S, K, T, r, sigma, option_type)
        diff = price - option_price
        
        if abs(diff) < tolerance:
            return sigma
        
        # Vega (derivative of price with respect to sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            break
        
        # Newton-Raphson update
        sigma = sigma - diff / vega
        sigma = max(0.01, min(5.0, sigma))  # Keep sigma in reasonable range
    
    return sigma


# ==================== POSITION SIZING ====================

def kelly_criterion(
    win_probability: float,
    win_loss_ratio: float,
    max_allocation: float = 0.25
) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Formula: f* = (p * b - q) / b
    where:
        f* = fraction of capital to bet
        p = win probability
        q = loss probability (1 - p)
        b = win/loss ratio
    
    Args:
        win_probability: Probability of winning trade (0-1)
        win_loss_ratio: Ratio of average win to average loss
        max_allocation: Maximum position size cap (default 25%)
        
    Returns:
        Optimal fraction of capital to allocate (0-max_allocation)
        
    Example:
        >>> # 60% win rate, avg win = 1.5x avg loss
        >>> kelly = kelly_criterion(0.6, 1.5)
        >>> print(f"Optimal position size: {kelly:.1%}")
    """
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be between 0 and 1")
    
    q = 1 - win_probability
    
    # Kelly formula
    kelly_fraction = (win_probability * win_loss_ratio - q) / win_loss_ratio
    
    # Apply cap and floor
    kelly_fraction = max(0.0, min(kelly_fraction, max_allocation))
    
    # Fractional Kelly (more conservative)
    fractional_kelly = kelly_fraction * 0.5  # Half Kelly for safety
    
    return fractional_kelly


def volatility_position_sizer(
    base_size: float,
    current_volatility: float,
    target_volatility: float = 0.15
) -> float:
    """
    Scale position size inversely with volatility.
    
    Goal: Maintain constant risk exposure regardless of volatility.
    
    Args:
        base_size: Base position size (e.g., number of shares)
        current_volatility: Current market volatility (annualized)
        target_volatility: Target risk level (default 15%)
        
    Returns:
        Adjusted position size
        
    Example:
        >>> # If volatility doubles, position size halves
        >>> size = volatility_position_sizer(100, current_volatility=0.30, target_volatility=0.15)
        >>> print(f"Adjusted size: {size} shares")  # ~50 shares
    """
    if current_volatility <= 0:
        return base_size
    
    # Inverse scaling: higher vol = smaller size
    vol_ratio = target_volatility / current_volatility
    adjusted_size = base_size * vol_ratio
    
    # Apply reasonable bounds
    adjusted_size = max(base_size * 0.2, min(adjusted_size, base_size * 3.0))
    
    return adjusted_size


def optimal_f(
    returns: Union[List[float], pd.Series],
    num_steps: int = 100
) -> float:
    """
    Find optimal fixed fraction using Ralph Vince's method.
    
    Maximizes geometric growth by testing different allocation fractions.
    
    Args:
        returns: Historical returns (as fractions, e.g., 0.05 for 5% gain)
        num_steps: Number of fractions to test
        
    Returns:
        Optimal fraction (0-1)
    """
    returns_array = np.array(returns)
    
    best_f = 0.0
    best_geom_mean = -np.inf
    
    for f in np.linspace(0.01, 0.99, num_steps):
        # Calculate terminal wealth for each return
        equity_multiples = 1 + f * returns_array
        
        # Avoid bankruptcy
        if np.any(equity_multiples <= 0):
            continue
        
        # Geometric mean
        geom_mean = np.prod(equity_multiples) ** (1 / len(equity_multiples))
        
        if geom_mean > best_geom_mean:
            best_geom_mean = geom_mean
            best_f = f
    
    return best_f


# ==================== RISK METRICS ====================

def calculate_sharpe_ratio(
    returns: Union[List[float], pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe = (Mean Return - Risk Free) / Std Dev of Returns
    
    Args:
        returns: Periodic returns (e.g., daily)
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: 252 for daily, 12 for monthly
        
    Returns:
        Annualized Sharpe ratio
    """
    returns_array = np.array(returns)
    
    if len(returns_array) < 2:
        return 0.0
    
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_mean = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    sharpe = (annual_mean - risk_free_rate) / annual_std
    
    return sharpe


def calculate_max_drawdown(
    equity_curve: Union[List[float], pd.Series]
) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    MDD = max(peak - trough) / peak
    
    Args:
        equity_curve: Portfolio values over time
        
    Returns:
        Maximum drawdown (positive number, e.g., 0.25 for 25% drawdown)
    """
    equity_array = np.array(equity_curve)
    
    if len(equity_array) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown at each point
    drawdown = (running_max - equity_array) / running_max
    
    # Maximum drawdown
    max_dd = np.max(drawdown)
    
    return max_dd


def calculate_sortino_ratio(
    returns: Union[List[float], pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (like Sharpe but only penalizes downside volatility).
    
    Args:
        returns: Periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Annualization factor
        
    Returns:
        Annualized Sortino ratio
    """
    returns_array = np.array(returns)
    
    if len(returns_array) < 2:
        return 0.0
    
    mean_return = np.mean(returns_array)
    
    # Downside deviation (only negative returns)
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0:
        downside_std = 0.01  # Avoid division by zero
    else:
        downside_std = np.std(downside_returns, ddof=1)
    
    # Annualize
    annual_mean = mean_return * periods_per_year
    annual_downside_std = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annual_mean - risk_free_rate) / annual_downside_std
    
    return sortino


def calculate_calmar_ratio(
    returns: Union[List[float], pd.Series],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Higher is better - measures return per unit of drawdown risk.
    
    Args:
        returns: Periodic returns
        periods_per_year: Annualization factor
        
    Returns:
        Calmar ratio
    """
    returns_array = np.array(returns)
    
    if len(returns_array) < 2:
        return 0.0
    
    # Annualized return
    total_return = np.prod(1 + returns_array) - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(returns_array)) - 1
    
    # Max drawdown
    equity_curve = np.cumprod(1 + returns_array)
    max_dd = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    calmar = annual_return / max_dd
    
    return calmar


# ==================== HELPER FUNCTIONS ====================

def annualized_volatility(
    returns: Union[List[float], pd.Series],
    periods_per_year: int = 252
) -> float:
    """Calculate annualized volatility from periodic returns."""
    returns_array = np.array(returns)
    if len(returns_array) < 2:
        return 0.0
    
    period_vol = np.std(returns_array, ddof=1)
    annual_vol = period_vol * np.sqrt(periods_per_year)
    
    return annual_vol


def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Useful for detecting strategy degradation over time.
    """
    def sharpe_func(r):
        if len(r) < 2:
            return np.nan
        return calculate_sharpe_ratio(r, risk_free_rate, periods_per_year)
    
    rolling_sharpe_values = returns.rolling(window=window).apply(sharpe_func, raw=False)
    
    return rolling_sharpe_values
