import { toast as sonnerToast } from 'sonner';

/**
 * Custom toast hook with predefined styles and behaviors
 */
export function useToast() {
  return {
    success: (message: string) => {
      sonnerToast.success(message);
    },
    error: (message: string) => {
      sonnerToast.error(message);
    },
    info: (message: string) => {
      sonnerToast.info(message);
    },
    warning: (message: string) => {
      sonnerToast.warning(message);
    },
    loading: (message: string) => {
      return sonnerToast.loading(message);
    },
    promise: <T,>(
      promise: Promise<T>,
      {
        loading,
        success,
        error,
      }: {
        loading: string;
        success: string | ((data: T) => string);
        error: string | ((error: Error) => string);
      }
    ) => {
      return sonnerToast.promise(promise, {
        loading,
        success,
        error,
      });
    },
    dismiss: (toastId?: string | number) => {
      sonnerToast.dismiss(toastId);
    },
  };
}
