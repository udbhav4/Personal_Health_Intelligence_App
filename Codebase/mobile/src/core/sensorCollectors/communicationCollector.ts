/**
 * Communication collector — derives communication node from Android CallLog and SMS.
 *
 * Window: last SENSING_INTERVAL_MINUTES minutes (passed in from passiveSensing).
 *
 * Produces three source columns per collection cycle:
 *   call_count           — number of calls in the window (all types: in/out/missed)
 *   call_duration_total  — total call talk time in minutes
 *   sms_count            — number of SMS messages (inbox + sent)
 *
 * Permissions required (runtime, declared in AndroidManifest.xml):
 *   READ_CALL_LOG
 *   READ_SMS
 * Both must be granted; if either is absent the collector returns null.
 *
 * Discretization — population z-score regime (Step 5 of DBN retraining plan):
 *   1. Compute population z-score: z = (raw - pop_mean) / pop_std
 *   2. Apply bin_edges (in population z-score space) via discretizeByEdges
 *   3. Use personal history ONLY to adjust confidence — never to classify
 *
 * Confidence logic:
 *   < 7 distinct calendar days of personal history in last 30 days → 0.65
 *   ≥ 7 days, personal z-score → same bin as population z-score           → 0.90
 *   ≥ 7 days, personal z-score → different bin from population z-score     → 0.75
 *   (population discretized_value always wins; confidence only communicates agreement)
 *
 * Returns null when either permission is not granted.
 * All three results are returned together or not at all (permission is shared).
 *
 * Android only — no iOS stubs needed.
 */

import { NativeModules, Platform } from 'react-native';
import type { DB } from '@op-engineering/op-sqlite';
import type { CollectorResult } from './types';
import { POPULATION_NORM_STATS, discretizeByEdges } from './populationNormStats';

// ── Constants ──────────────────────────────────────────────────────────────────

const COLD_START_DAYS     = 7;
const ROLLING_WINDOW_DAYS = 30;

// Confidence tiers
const CONF_NO_HISTORY = 0.65;
const CONF_AGREE      = 0.90;
const CONF_DISAGREE   = 0.75;

// ── Bin edges and state labels (from retrained feature_node_config.json) ───────
// All edges are in population z-score space — must apply population z-score first.
//
// call_count: 4 states, 3 edges
//   z <= 1.00608                  → 'low'
//   1.00608 < z <= 4.518805       → 'moderate_low'
//   4.518805 < z <= 12.959386     → 'moderate_high'
//   z > 12.959386                 → 'high'
//
// call_duration_total: 2 states, 1 edge
//   z <= 7.70951  → 'low'
//   z > 7.70951   → 'high'
//
// sms_count: 2 states, 1 edge
//   z <= 3.826932 → 'low'
//   z > 3.826932  → 'high'

const CALL_COUNT_BIN_EDGES    = [1.00608, 4.518805, 12.959386];
const CALL_COUNT_STATE_LABELS = ['low', 'moderate_low', 'moderate_high', 'high'];

const CALL_DURATION_BIN_EDGES    = [7.70951];
const CALL_DURATION_STATE_LABELS = ['low', 'high'];

const SMS_COUNT_BIN_EDGES    = [3.826932];
const SMS_COUNT_STATE_LABELS = ['low', 'high'];

// ── Native module interface ────────────────────────────────────────────────────

interface CommunicationResult {
  hasPermission:       boolean;
  callCount:           number;
  callDurationMinutes: number;
  smsCount:            number;
}

interface SensorNativeModuleInterface {
  getCommunicationStats(windowMs: number): Promise<CommunicationResult>;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function computeMeanStd(values: number[]): { mean: number; std: number } {
  const n        = values.length;
  const mean     = values.reduce((s, v) => s + v, 0) / n;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  return { mean, std: Math.sqrt(variance) };
}

interface HistoryRow { raw_value: number; date: string }

/**
 * Query rolling personal history for one (communication, source_column) pair.
 * Returns raw_value array and count of distinct calendar days within the window.
 */
function queryCommunicationHistory(
  db: DB,
  sourceColumn: string,
  rollingDays: number,
): { values: number[]; distinctDays: number } {
  const cutoff = new Date(Date.now() - rollingDays * 86_400_000)
    .toLocaleDateString('sv'); // YYYY-MM-DD

  const rows = db.executeSync(
    `SELECT raw_value, date
     FROM   sensor_windows
     WHERE  node_name     = 'communication'
       AND  source_column = ?
       AND  raw_value     IS NOT NULL
       AND  date          >= ?
     ORDER  BY date DESC, snapshot_time DESC`,
    [sourceColumn, cutoff],
  ).rows as unknown as HistoryRow[];

  const values       = rows.map(r => Number(r.raw_value));
  const distinctDays = new Set(rows.map(r => r.date)).size;
  return { values, distinctDays };
}

/**
 * Classify one communication metric using population z-score and compute
 * confidence from personal history agreement.
 *
 * Algorithm:
 *   1. Compute population z-score and discretize → discretized_value (final, immutable)
 *   2. If < 7 distinct days of personal history → confidence = CONF_NO_HISTORY
 *   3. If personal std is effectively 0 (flat history) → confidence = CONF_NO_HISTORY
 *   4. Otherwise compute personal z-score, discretize with same edges
 *      → match → CONF_AGREE, mismatch → CONF_DISAGREE
 *
 * @param rawValue    - current raw sensor reading
 * @param colKey      - key into POPULATION_NORM_STATS
 * @param binEdges    - bin edges in population z-score space (any length ≥ 1)
 * @param stateLabels - corresponding state labels (length = binEdges.length + 1)
 * @param history     - personal rolling history for this column
 */
function classifyCommCol(
  rawValue:    number,
  colKey:      string,
  binEdges:    number[],
  stateLabels: string[],
  history:     { values: number[]; distinctDays: number },
): { discretized_value: string; confidence: number } {
  const { values, distinctDays } = history;

  // ── Step 1: population z-score classification (always the final answer) ───────
  const popStats = POPULATION_NORM_STATS[colKey];

  // Defensive guard: if pop_std is 0 or NaN, skip z-scoring, fall back to raw
  if (!popStats || popStats.std < 1e-9 || !isFinite(popStats.std)) {
    const discretized_value = discretizeByEdges(rawValue, binEdges, stateLabels);
    return { discretized_value, confidence: CONF_NO_HISTORY };
  }

  const zPop              = (rawValue - popStats.mean) / popStats.std;
  const discretized_value = discretizeByEdges(zPop, binEdges, stateLabels);

  // ── Step 2: check personal history for confidence adjustment ─────────────────
  if (distinctDays < COLD_START_DAYS || values.length === 0) {
    return { discretized_value, confidence: CONF_NO_HISTORY };
  }

  const { mean: personalMean, std: personalStd } = computeMeanStd(values);

  // Flat personal history → no personal signal
  if (personalStd < 1e-9) {
    return { discretized_value, confidence: CONF_NO_HISTORY };
  }

  const zPersonal           = (rawValue - personalMean) / personalStd;
  const personalDiscretized = discretizeByEdges(zPersonal, binEdges, stateLabels);

  const confidence = personalDiscretized === discretized_value ? CONF_AGREE : CONF_DISAGREE;
  return { discretized_value, confidence };
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * Collect communication stats for the last `windowMinutes` minutes.
 * Returns three CollectorResults (one per source_column) or null if READ_CALL_LOG
 * or READ_SMS permission is not granted.
 *
 * @param db            - op-sqlite DB handle (for personal history lookup)
 * @param windowMinutes - sensing window size in minutes
 */
export async function collectCommunication(
  db: DB,
  windowMinutes: number,
): Promise<CollectorResult[] | null> {
  // Android only — skip silently on other platforms
  if (Platform.OS !== 'android') return null;

  const module = NativeModules.SensorNativeModule as SensorNativeModuleInterface | undefined;
  if (!module) return null;

  let stats: CommunicationResult;
  try {
    stats = await module.getCommunicationStats(windowMinutes * 60_000);
  } catch {
    return null;
  }

  if (!stats.hasPermission) return null;

  // ── Personal history lookups ─────────────────────────────────────────────────
  const callCountHistory    = queryCommunicationHistory(db, 'call_count',         ROLLING_WINDOW_DAYS);
  const callDurationHistory = queryCommunicationHistory(db, 'call_duration_total', ROLLING_WINDOW_DAYS);
  const smsCountHistory     = queryCommunicationHistory(db, 'sms_count',           ROLLING_WINDOW_DAYS);

  // ── Classify each column ─────────────────────────────────────────────────────
  const callCountDis = classifyCommCol(
    stats.callCount,
    'call_count',
    CALL_COUNT_BIN_EDGES,
    CALL_COUNT_STATE_LABELS,
    callCountHistory,
  );
  const callDurationDis = classifyCommCol(
    stats.callDurationMinutes,
    'call_duration_total',
    CALL_DURATION_BIN_EDGES,
    CALL_DURATION_STATE_LABELS,
    callDurationHistory,
  );
  const smsCountDis = classifyCommCol(
    stats.smsCount,
    'sms_count',
    SMS_COUNT_BIN_EDGES,
    SMS_COUNT_STATE_LABELS,
    smsCountHistory,
  );

  return [
    {
      node_name:         'communication',
      source_column:     'call_count',
      data_source:       'call_log',
      raw_value:         stats.callCount,
      raw_unit:          'count',
      discretized_value: callCountDis.discretized_value,
      confidence:        callCountDis.confidence,
    },
    {
      node_name:         'communication',
      source_column:     'call_duration_total',
      data_source:       'call_log',
      raw_value:         stats.callDurationMinutes,
      raw_unit:          'minutes',
      discretized_value: callDurationDis.discretized_value,
      confidence:        callDurationDis.confidence,
    },
    {
      node_name:         'communication',
      source_column:     'sms_count',
      data_source:       'telephony_sms',
      raw_value:         stats.smsCount,
      raw_unit:          'count',
      discretized_value: smsCountDis.discretized_value,
      confidence:        smsCountDis.confidence,
    },
  ];
}
