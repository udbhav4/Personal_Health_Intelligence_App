/**
 * Activity collector — derives activity node from expo-pedometer step count.
 *
 * Window: last SENSING_INTERVAL_MINUTES minutes (passed in from passiveSensing).
 * Metric: active_ratio = min(stepsInWindow / 1500, 1.0)
 *   • 1500 = fully active 15-min window at ~100 steps/min
 *
 * Discretization — clinical threshold regime (Step 5 of DBN retraining plan):
 *   active_ratio is classified using a fixed clinical threshold, NOT population z-score.
 *   Threshold updated from 0.33 → 0.25 after retraining (new bin: [0.0, 0.25, 1.01]).
 *
 *   Classification (immutable):
 *     active_ratio <= 0.25 → 'low'   (pd.cut right=True semantics)
 *     active_ratio >  0.25 → 'high'
 *
 * Confidence logic (personal history, last 30 days):
 *   < 7 distinct calendar days of history → 0.65
 *   ≥ 7 days, majority (> 50%) of historical ratios fall in same clinical bin → 0.90
 *   ≥ 7 days, majority fall in different bin                                  → 0.75
 *
 * IMPORTANT — raw_value stored in DB is stepsInWindow (integer steps), NOT active_ratio.
 * History rows must be converted: Math.min(v / 1500, 1.0) before computing bin majority.
 *
 * Returns null when pedometer permission is denied or hardware unavailable.
 */

import { Pedometer } from 'expo-sensors';
import type { DB } from '@op-engineering/op-sqlite';
import type { CollectorResult } from './types';

// ── Constants ──────────────────────────────────────────────────────────────────

const FULL_ACTIVE_STEPS   = 1500;  // steps = 100% active in one 15-min window
const COLD_START_DAYS     = 7;     // minimum distinct days before using personal history
const ROLLING_WINDOW_DAYS = 30;    // look-back window for personal history

// Clinical threshold: active_ratio <= CLINICAL_THRESHOLD → 'low', else → 'high'
// Updated from 0.33 → 0.25 after DBN retraining (new bin_edges=[0.0, 0.25, 1.01])
const CLINICAL_THRESHOLD  = 0.25;

// Confidence tiers
const CONF_NO_HISTORY = 0.65;
const CONF_AGREE      = 0.90;
const CONF_DISAGREE   = 0.75;

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Classify an active_ratio value using the clinical threshold.
 * Matches pd.cut(right=True) semantics: value <= threshold → lower bin.
 */
function clinicalBin(ratio: number): string {
  return ratio <= CLINICAL_THRESHOLD ? 'low' : 'high';
}

/**
 * Query rolling personal history of active_ratio for the 'activity' node.
 *
 * NOTE: raw_value in the DB is stored as stepsInWindow (integer steps).
 * Conversion to active_ratio is applied AFTER the query.
 *
 * Returns:
 *   - ratios: active_ratio values (already converted from steps)
 *   - distinctDays: count of distinct calendar days in the result set
 */
function queryActivityHistory(
  db: DB,
  rollingDays: number,
): { ratios: number[]; distinctDays: number } {
  interface Row { raw_value: number; date: string }

  const cutoff = new Date(Date.now() - rollingDays * 86_400_000)
    .toLocaleDateString('sv'); // YYYY-MM-DD

  const rows = db.executeSync(
    `SELECT raw_value, date
     FROM   sensor_windows
     WHERE  node_name     = 'activity'
       AND  source_column = 'active_ratio'
       AND  raw_value     IS NOT NULL
       AND  date          >= ?
     ORDER  BY date DESC, snapshot_time DESC`,
    [cutoff],
  ).rows as unknown as Row[];

  // raw_value is stepsInWindow — convert to active_ratio
  const ratios       = rows.map(r => Math.min(Number(r.raw_value) / FULL_ACTIVE_STEPS, 1.0));
  const distinctDays = new Set(rows.map(r => r.date)).size;
  return { ratios, distinctDays };
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * Collect step count for the last `windowMinutes` minutes and return an
 * activity CollectorResult, or null if pedometer is unavailable.
 *
 * @param db            - op-sqlite DB handle (for personal history lookup)
 * @param windowMinutes - sensing window size in minutes
 */
export async function collectActivity(
  db: DB,
  windowMinutes: number,
): Promise<CollectorResult | null> {
  // Permission check
  const { granted } = await Pedometer.requestPermissionsAsync();
  if (!granted) return null;

  // Availability check
  const available = await Pedometer.isAvailableAsync();
  if (!available) return null;

  // Step count for the current window
  const windowEnd   = new Date();
  const windowStart = new Date(windowEnd.getTime() - windowMinutes * 60_000);

  let stepsInWindow: number;
  try {
    const result  = await Pedometer.getStepCountAsync(windowStart, windowEnd);
    stepsInWindow = result.steps;
  } catch {
    // Pedometer can throw on first call on some devices — treat as unavailable
    return null;
  }

  const activeRatio = Math.min(stepsInWindow / FULL_ACTIVE_STEPS, 1.0);

  // ── Classification: clinical threshold (immutable) ───────────────────────────
  const discretized_value = clinicalBin(activeRatio);
  const currentBin        = discretized_value; // alias for clarity below

  // ── Confidence: personal history majority vote ───────────────────────────────
  const { ratios: historyRatios, distinctDays } = queryActivityHistory(db, ROLLING_WINDOW_DAYS);

  let confidence: number;

  if (distinctDays < COLD_START_DAYS || historyRatios.length === 0) {
    // Not enough personal history
    confidence = CONF_NO_HISTORY;
  } else {
    // Count how many historical ratios fall in the same clinical bin as current
    const sameCount = historyRatios.filter(r => clinicalBin(r) === currentBin).length;
    const majority  = sameCount / historyRatios.length > 0.5;
    confidence      = majority ? CONF_AGREE : CONF_DISAGREE;
  }

  return {
    node_name:         'activity',
    source_column:     'active_ratio',
    data_source:       'pedometer',
    raw_value:         stepsInWindow,  // integer step count (raw sensor reading)
    raw_unit:          'steps',
    discretized_value,
    confidence,
  };
}
