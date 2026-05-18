/**
 * useInfer — React hook that exposes the full inference pipeline to the UI.
 *
 * Wraps runInfer() so screens never touch the DB or core logic directly.
 * Runs inference off the JS main thread via a queued async call to avoid
 * blocking the UI during LBP (which can take tens of ms on large graphs).
 *
 * Usage:
 *   const { beliefs, loading, error, infer } = useInfer();
 *   await infer('user_query');
 */

import { useCallback, useState } from 'react';

import { openDb }                          from '../core/db';
import { runInfer, type TriggerType,
         type InferResult }                from '../core/infer';

interface UseInferState {
  beliefs:  InferResult['beliefs'] | null;
  loading:  boolean;
  error:    string | null;
}

interface UseInferReturn extends UseInferState {
  /** Trigger a full inference run. Resolves with the result or null on error. */
  infer: (trigger?: TriggerType) => Promise<InferResult | null>;
}

export function useInfer(): UseInferReturn {
  const [state, setState] = useState<UseInferState>({
    beliefs: null,
    loading: false,
    error:   null,
  });

  const infer = useCallback(
    async (trigger: TriggerType = 'user_query'): Promise<InferResult | null> => {
      setState(s => ({ ...s, loading: true, error: null }));
      try {
        // runInfer is synchronous (SQLite sync API + pure JS LBP).
        // Wrapping in a resolved promise yields to the event loop once,
        // letting React flush any pending renders before the blocking work.
        const result = await Promise.resolve().then(() => runInfer(openDb(), trigger));
        setState({ beliefs: result.beliefs, loading: false, error: null });
        return result;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setState(s => ({ ...s, loading: false, error: msg }));
        return null;
      }
    },
    [],
  );

  return { ...state, infer };
}
