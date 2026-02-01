'use client';

import { useState, useEffect } from 'react';
import { checkBackendConnection } from '@/services/api';

interface ApiStatus {
  connected: boolean;
  checking: boolean;
  lastCheck: Date | null;
}

export function useApiStatus() {
  const [status, setStatus] = useState<ApiStatus>({
    connected: false,
    checking: true,
    lastCheck: null,
  });

  const checkConnection = async () => {
    setStatus((prev) => ({ ...prev, checking: true }));
    const isConnected = await checkBackendConnection();
    setStatus({
      connected: isConnected,
      checking: false,
      lastCheck: new Date(),
    });
  };

  useEffect(() => {
    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return { ...status, checkConnection };
}
