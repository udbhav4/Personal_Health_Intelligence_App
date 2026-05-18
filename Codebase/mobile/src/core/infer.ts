/**
 * Full inference pipeline — on-device port of POST /infer in codebase/backend/server.py.
 *
 * Steps (identical to Python):
 *   1. Pull evidence from DB (user_data_sensorless + sensor_windows)
 *   2. Load t-1 beliefs from inference_snapshots → apply inter-slice transitions
 *   3. Merge sensorless soft-evidence + temporal priors
 *   4. Run LBP
 *   5. Persist inference_snapshots row
 *   6. Return beliefs + metadata
 *
 * Usage:
 *   import { runInfer } from './infer';
 *   const result = runInfer(db, 'user_query');
 */

import type { DB } from '@op-engineering/op-sqlite';

import { buildDbnEvidence, type DbnEvidence }          from './evidenceLayer';
import { runLBP, applyInterSlice, formatBeliefs,
         type BeliefResult, type PriorFactors }         from './inferenceEngine';

// ── Types ─────────────────────────────────────────────────────────────────────

export type TriggerType = 'scheduled' | 'sensor_event' | 'user_query';

export interface InferResult {
  beliefs:           BeliefResult;
  node_confidences:  Record<string, number>;
  node_data_sources: Record<string, string>;
  snapshot_time:     string;
  summary_line:      string | null;
}

// ── Temporal prior helper ─────────────────────────────────────────────────────

/**
 * Load t-1 beliefs from inference_snapshots, apply inter-slice matrices.
 * Returns {} when no prior snapshot exists.
 * Mirrors _temporal_priors_from_last_snapshot() in server.py.
 */
function temporalPriorsFromLastSnapshot(db: DB): PriorFactors {
  const row = db.executeSync(
    `SELECT dbn_beliefs
     FROM   inference_snapshots
     ORDER  BY date DESC, snapshot_time DESC
     LIMIT  1`,
  ).rows[0] as { dbn_beliefs: string } | undefined;
  if (!row?.dbn_beliefs) return {};

  let prevBeliefs: BeliefResult;
  try {
    prevBeliefs = JSON.parse(row.dbn_beliefs) as BeliefResult;
  } catch {
    return {};
  }

  return applyInterSlice(prevBeliefs);
}

// ── Prior-factor merge helper ─────────────────────────────────────────────────

/**
 * Merge sensorless soft-evidence and temporal priors into one prior_factors dict.
 *
 * For nodes in both: element-wise multiply then normalise (both are unary factors).
 * For nodes in only one source: pass through unchanged.
 * Mirrors _merge_prior_factors() in server.py.
 */
function mergePriorFactors(
  sensorlessPF: Record<string, number[]>,
  temporalPF:   PriorFactors,
): PriorFactors {
  const allNodes = new Set([...Object.keys(sensorlessPF), ...Object.keys(temporalPF)]);
  const merged: PriorFactors = {};

  for (const node of allNodes) {
    const hasSensorless = node in sensorlessPF;
    const hasTemporal   = node in temporalPF;

    if (hasSensorless && hasTemporal) {
      // Element-wise multiply: both are unary factors, product = joint belief
      const combined = sensorlessPF[node].map((v, i) => v * temporalPF[node][i]);
      const s        = combined.reduce((a, b) => a + b, 0);
      const k        = combined.length;
      merged[node]   = s > 0 ? combined.map(v => v / s) : Array(k).fill(1 / k);
    } else if (hasSensorless) {
      merged[node] = sensorlessPF[node];
    } else {
      merged[node] = temporalPF[node];
    }
  }

  return merged;
}

// ── Main pipeline ─────────────────────────────────────────────────────────────

/**
 * Run the full inference pipeline synchronously.
 * Returns formatted beliefs + metadata, same shape as InferResponse in server.py.
 */
export function runInfer(db: DB, triggerType: TriggerType): InferResult {
  const now          = new Date();
  const dateStr      = now.toLocaleDateString('sv');                               // YYYY-MM-DD local
  const snapshotTime = now.toLocaleString('sv').replace(' ', 'T').slice(0, 19);   // YYYY-MM-DDTHH:MM:SS local

  // Step 1 — evidence from DB
  const evData: DbnEvidence = buildDbnEvidence(db);

  // Step 2 — temporal priors from t-1 snapshot
  const temporalPF = temporalPriorsFromLastSnapshot(db);

  // Step 3 — merge sensorless soft-evidence + temporal priors
  const mergedPF = mergePriorFactors(evData.prior_factors, temporalPF);

  // Step 4 — validate hard evidence (drop unknown nodes/states), run LBP
  // Unknown node/state validation happens inside runLBP via the state index lookup;
  // invalid states default to index 0 (same defensive behaviour as server.py).
  const rawBeliefs = runLBP(
    evData.evidence,
    Object.keys(mergedPF).length > 0 ? mergedPF : undefined,
  );
  const beliefs = formatBeliefs(rawBeliefs);

  // Step 5 — persist snapshot (INSERT OR REPLACE mirrors Python's upsert pattern)
  db.executeSync(
    `INSERT OR REPLACE INTO inference_snapshots
       (date, snapshot_time, trigger_type,
        prior_beliefs, sensor_snapshot, sensorless_snapshot,
        dbn_beliefs, node_confidences, node_data_sources)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      dateStr,
      snapshotTime,
      triggerType,
      // prior_beliefs stored as {node: [p0,...]} — same format as Python
      JSON.stringify(
        Object.fromEntries(
          Object.entries(temporalPF).map(([n, arr]) => [n, Array.from(arr)]),
        ),
      ),
      JSON.stringify(evData.sensor_snapshot),
      JSON.stringify(evData.sensorless_snapshot),
      JSON.stringify(beliefs),
      JSON.stringify(evData.node_confidences),
      JSON.stringify(evData.node_data_sources),
    ],
  );

  // Step 6 — return result (summary_line populated by SLM layer later)
  return {
    beliefs,
    node_confidences:  evData.node_confidences,
    node_data_sources: evData.node_data_sources,
    snapshot_time:     snapshotTime,
    summary_line:      null,
  };
}
