/**
 * Reads the latest inference_snapshots row from SQLite and surfaces it as
 * `beliefs` in AppContext. Call after any chat turn completes so the
 * dashboard rings update automatically.
 */

import { useCallback } from 'react';
import { useAppContext } from '../core/AppContext';
import type { BeliefResult } from '../core/inferenceEngine';

export function useBeliefRefresh() {
  const { db, setBeliefs } = useAppContext();

  const refresh = useCallback(() => {
    if (!db) return;
    const row = db.executeSync(
      `SELECT dbn_beliefs FROM inference_snapshots ORDER BY date DESC, snapshot_time DESC LIMIT 1`,
    ).rows[0] as { dbn_beliefs: string } | undefined;

    if (!row?.dbn_beliefs) return;
    try {
      const beliefs = JSON.parse(row.dbn_beliefs) as BeliefResult;
      setBeliefs(beliefs);
    } catch { /* corrupt snapshot — skip */ }
  }, [db, setBeliefs]);

  return refresh;
}
