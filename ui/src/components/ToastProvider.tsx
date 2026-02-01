'use client';

import { Toaster as SonnerToaster } from 'sonner';

/**
 * Toast Provider Component
 * Wraps the application to provide toast notifications
 */
export function ToastProvider() {
  return (
    <SonnerToaster
      position="top-right"
      richColors
      expand={false}
      duration={4000}
      toastOptions={{
        style: {
          background: 'rgba(17, 24, 39, 0.95)',
          border: '1px solid rgba(75, 85, 99, 0.3)',
          color: '#f3f4f6',
        },
        className: 'backdrop-blur-sm',
      }}
    />
  );
}
