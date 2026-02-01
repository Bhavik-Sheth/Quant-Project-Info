'use client';

import { useApiStatus } from '@/hooks/useApiStatus';

/**
 * API Status Indicator Component
 * Shows connection status to the backend API
 */
export function ApiStatusIndicator() {
  const { connected, checking, lastCheck, checkConnection } = useApiStatus();

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-lg px-4 py-2 shadow-lg">
        <div className="flex items-center gap-3">
          {/* Status Indicator */}
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                checking
                  ? 'bg-yellow-500 animate-pulse'
                  : connected
                  ? 'bg-green-500'
                  : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-gray-300">
              {checking ? 'Checking...' : connected ? 'Backend Connected' : 'Backend Offline'}
            </span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={checkConnection}
            disabled={checking}
            className="text-xs text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50"
            title="Check connection"
          >
            <svg
              className={`w-4 h-4 ${checking ? 'animate-spin' : ''}`}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>

        {/* Last Check Time */}
        {lastCheck && (
          <div className="text-xs text-gray-500 mt-1">
            Last check: {lastCheck.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}
