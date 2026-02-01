// import React from 'react'

//  function BacktestPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Backtesting</h1>
//       <p className="text-gray-400">
//         Simulate strategy performance on historical data.
//       </p>
//     </div>
//   );
// }

// export default BacktestPage


"use client";

import { useState } from "react";
import { api } from "@/services/api";
import { useToast } from "@/hooks/useToast";
import type { BacktestResult } from "@/types/api";
import ClientShell from "@/components/ClientShell";

function BacktestPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  // Form state
  const [symbol, setSymbol] = useState("BTCUSD");
  const [timeframe, setTimeframe] = useState("1d");
  const [strategy, setStrategy] = useState("sma_crossover");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");

  const runBacktest = async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);

    try {
      const result = await api.runBacktest({
        symbol,
        timeframe,
        strategy,
        initial_capital: initialCapital,
        start_date: startDate,
        end_date: endDate,
      });

      setResults(result);
      toast.success("Backtest completed successfully!");
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || "Backtest failed";
      setError(errorMsg);
      toast.error(`Backtest failed: ${errorMsg}`);
      console.error("Backtest error:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <ClientShell>
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold">Backtesting</h1>

        {/* Configuration Form */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Backtest Configuration</h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
                placeholder="BTCUSD"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Timeframe</label>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="1d">1 Day</option>
                <option value="4h">4 Hours</option>
                <option value="1h">1 Hour</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="sma_crossover">SMA Crossover</option>
                <option value="mean_reversion">Mean Reversion</option>
                <option value="multi_agent">Multi-Agent</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Initial Capital ($)</label>
              <input
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <button
            onClick={runBacktest}
            disabled={isRunning}
            className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded transition-colors font-medium"
          >
            {isRunning ? "Running Backtest..." : "Run Backtest"}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Results */}
        {results && (
          <>
            {/* Metrics */}
            <div className="grid grid-cols-4 gap-4">
              <Metric label="Total Return" value={`${(results.total_return * 100).toFixed(2)}%`} />
              <Metric label="Sharpe Ratio" value={results.sharpe_ratio.toFixed(2)} />
              <Metric label="Max Drawdown" value={`${(results.max_drawdown * 100).toFixed(2)}%`} />
              <Metric label="Win Rate" value={`${(results.win_rate * 100).toFixed(2)}%`} />
              <Metric label="Total Trades" value={results.total_trades} />
            </div>

            {/* Equity Curve */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h2 className="text-lg font-semibold mb-2">Equity Curve</h2>
              {results.equity_curve && results.equity_curve.length > 0 ? (
                <div className="h-64 flex items-center justify-center text-gray-400">
                  <p>Equity curve visualization - {results.equity_curve.length} data points</p>
                  {/* TODO: Add actual chart component here */}
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-gray-400">
                  No equity curve data available
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </ClientShell>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center">
      <div className="text-sm text-gray-400">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}

export default BacktestPage;
