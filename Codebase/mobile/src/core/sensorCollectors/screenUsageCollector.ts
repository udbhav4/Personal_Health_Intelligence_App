/**
 * Screen-usage collector — derives screen_usage node from Android UsageStatsManager.
 *
 * Window: last SENSING_INTERVAL_MINUTES minutes (passed in from passiveSensing).
 *
 * Produces three source columns per collection cycle:
 *   screen_time_window_minutes — total foreground app time
 *   dark_window_minutes        — total non-interactive (screen-off/locked) time
 *   unlocked_window_minutes    — total time keyguard was dismissed (actively unlocked)
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
 * Returns null when PACKAGE_USAGE_STATS permission is not granted.
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
const CONF_NO_HISTORY  = 0.65;
const CONF_AGREE       = 0.90;
const CONF_DISAGREE    = 0.75;

// ── Bin edges and state labels (from retrained feature_node_config.json) ───────
// All edges are in population z-score space — must apply population z-score first.

const SCREEN_TIME_BIN_EDGES    = [0.050345, 1.340537];
const SCREEN_TIME_STATE_LABELS = ['low', 'moderate', 'high'];

const DARK_BIN_EDGES           = [-0.073285, 1.058989];
const DARK_STATE_LABELS        = ['low', 'moderate', 'high'];

const UNLOCKED_BIN_EDGES       = [-0.206671, 0.867869];
const UNLOCKED_STATE_LABELS    = ['low', 'moderate', 'high'];

// ── Native module interface ────────────────────────────────────────────────────

interface ScreenUsageResult {
  hasPermission:     boolean;
  screenTimeMinutes: number;
  darkMinutes:       number;
  unlockedMinutes:   number;
}

interface SensorNativeModuleInterface {
  getScreenUsageStats(windowMs: number): Promise<ScreenUsageResult>;
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
 * Query rolling personal history for one (screen_usage, source_column) pair.
 * Returns raw_value array and count of distinct calendar days within the window.
 */
function queryScreenHistory(
  db: DB,
  sourceColumn: string,
  rollingDays: number,
): { values: number[]; distinctDays: number } {
  const cutoff = new Date(Date.now() - rollingDays * 86_400_000)
    .toLocaleDateString('sv'); // YYYY-MM-DD

  const rows = db.executeSync(
    `SELECT raw_value, date
     FROM   sensor_windows
     WHERE  node_name     = 'screen_usage'
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
 * Classify one screen-usage metric using population z-score and compute
 * confidence from personal history agreement.
 *
 * Algorithm:
 *   1. Compute population z-score and discretize → discretized_value (final, immutable)
 *   2. If < 7 distinct days of personal history → confidence = CONF_NO_HISTORY
 *   3. If personal std is effectively 0 (flat history) → confidence = CONF_NO_HISTORY
 *   4. Otherwise compute personal z-score, discretize with same edges
 *      → match → CONF_AGREE, mismatch → CONF_DISAGREE
 *
 * @param rawValue    - current raw sensor reading in the metric's native unit
 * @param colKey      - key into POPULATION_NORM_STATS (e.g. 'screen_time_window_minutes')
 * @param binEdges    - bin edges in population z-score space
 * @param stateLabels - corresponding state labels
 * @param history     - personal rolling history for this column
 */
function classifyScreenCol(
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

  const zPersonal          = (rawValue - personalMean) / personalStd;
  const personalDiscretized = discretizeByEdges(zPersonal, binEdges, stateLabels);

  const confidence = personalDiscretized === discretized_value ? CONF_AGREE : CONF_DISAGREE;
  return { discretized_value, confidence };
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * Collect screen-usage stats for the last `windowMinutes` minutes.
 * Returns three CollectorResults (one per source_column) or null if the
 * PACKAGE_USAGE_STATS permission is not granted.
 *
 * @param db            - op-sqlite DB handle (for personal history lookup)
 * @param windowMinutes - sensing window size in minutes
 */
export async function collectScreenUsage(
  db: DB,
  windowMinutes: number,
): Promise<CollectorResult[] | null> {
  // Android only — skip silently on other platforms
  if (Platform.OS !== 'android') return null;

  const module = NativeModules.SensorNativeModule as SensorNativeModuleInterface | undefined;
  if (!module) return null;

  let stats: ScreenUsageResult;
  try {
    stats = await module.getScreenUsageStats(windowMinutes * 60_000);
  } catch {
    return null;
  }

  if (!stats.hasPermission) return null;

  // ── Personal history lookups ─────────────────────────────────────────────────
  const screenTimeHistory = queryScreenHistory(db, 'screen_time_window_minutes', ROLLING_WINDOW_DAYS);
  const darkHistory       = queryScreenHistory(db, 'dark_window_minutes',        ROLLING_WINDOW_DAYS);
  const unlockedHistory   = queryScreenHistory(db, 'unlocked_window_minutes',    ROLLING_WINDOW_DAYS);

  // ── Classify each column ─────────────────────────────────────────────────────
  const screenTimeDis = classifyScreenCol(
    stats.screenTimeMinutes,
    'screen_time_window_minutes',
    SCREEN_TIME_BIN_EDGES,
    SCREEN_TIME_STATE_LABELS,
    screenTimeHistory,
  );
  const darkDis = classifyScreenCol(
    stats.darkMinutes,
    'dark_window_minutes',
    DARK_BIN_EDGES,
    DARK_STATE_LABELS,
    darkHistory,
  );
  const unlockedDis = classifyScreenCol(
    stats.unlockedMinutes,
    'unlocked_window_minutes',
    UNLOCKED_BIN_EDGES,
    UNLOCKED_STATE_LABELS,
    unlockedHistory,
  );

  return [
    {
      node_name:         'screen_usage',
      source_column:     'screen_time_window_minutes',
      data_source:       'usage_stats_manager',
      raw_value:         stats.screenTimeMinutes,
      raw_unit:          'minutes',
      discretized_value: screenTimeDis.discretized_value,
      confidence:        screenTimeDis.confidence,
    },
    {
      node_name:         'screen_usage',
      source_column:     'dark_window_minutes',
      data_source:       'usage_stats_manager',
      raw_value:         stats.darkMinutes,
      raw_unit:          'minutes',
      discretized_value: darkDis.discretized_value,
      confidence:        darkDis.confidence,
    },
    {
      node_name:         'screen_usage',
      source_column:     'unlocked_window_minutes',
      data_source:       'usage_stats_manager',
      raw_value:         stats.unlockedMinutes,
      raw_unit:          'minutes',
      discretized_value: unlockedDis.discretized_value,
      confidence:        unlockedDis.confidence,
    },
  ];
}
