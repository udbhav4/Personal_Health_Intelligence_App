/**
 * Passive sensing orchestrator.
 *
 * Registers an expo-background-fetch task that fires every SENSING_INTERVAL_MINUTES
 * (iOS may throttle to longer intervals — the task is idempotent).
 *
 * Phase 1 sensors:
 *   time_of_day   — system clock (no permission required)
 *   activity      — expo-pedometer step count
 *   screen_usage  — Android UsageStatsManager (PACKAGE_USAGE_STATS app-op)
 *   communication — Android CallLog + SMS (READ_CALL_LOG + READ_SMS runtime permissions)
 *
 * Each Phase 1 collector returns CollectorResult[] | null (multiple source_columns per
 * node).  Null is returned silently when a permission is denied; the remaining
 * collectors still run.
 *
 * Prev-day aggregation: on the first window of each new calendar day, yesterday's
 * sensor_windows rows are aggregated per node and written back as 'prev_day_*' rows
 * for context continuity in evidence.
 */

import * as BackgroundFetch from 'expo-background-fetch';
import * as TaskManager     from 'expo-task-manager';
import type { DB }          from '@op-engineering/op-sqlite';

import { writeSensorReadings, type SensorWindowRow } from './db';
import { collectClock }         from './sensorCollectors/clockCollector';
import { collectActivity }      from './sensorCollectors/activityCollector';
import { collectScreenUsage }   from './sensorCollectors/screenUsageCollector';
import { collectCommunication } from './sensorCollectors/communicationCollector';
import type { CollectorResult } from './sensorCollectors/types';

// ── Config ─────────────────────────────────────────────────────────────────────

export const SENSING_INTERVAL_MINUTES = 15;
const TASK_NAME = 'PASSIVE_SENSING_TASK';

// ── Module-level DB reference (set by initPassiveSensing) ─────────────────────
// TaskManager callbacks cannot receive arguments, so we store the db handle here.
// This is a singleton pattern: only one DB connection exists in the app process.
let _db: DB | null = null;

// Track the last date we processed prev-day aggregation (to fire only once per day).
let _lastPrevDayDate: string | null = null;

// Track the last date we computed sleep hours (fires once per day after 11:00).
let _lastSleepDate: string | null = null;

// ── Datetime helpers ───────────────────────────────────────────────────────────

/** Returns local date as YYYY-MM-DD (device timezone). */
function localDate(): string {
  return new Date().toLocaleDateString('sv'); // 'sv' locale gives YYYY-MM-DD
}

/** Returns local time as HH:MM:SS (device timezone). */
function localTime(): string {
  return new Date().toLocaleTimeString('sv'); // 'sv' locale gives HH:MM:SS
}

/** Subtract minutes from a Date, return local HH:MM:SS. */
function localTimeMinutesAgo(minutes: number): string {
  return new Date(Date.now() - minutes * 60_000).toLocaleTimeString('sv');
}

/** Returns the previous calendar day as YYYY-MM-DD. */
function previousLocalDate(dateStr: string): string {
  const d = new Date(dateStr + 'T00:00:00');
  d.setDate(d.getDate() - 1);
  return d.toLocaleDateString('sv');
}

// ── Prev-day aggregation ───────────────────────────────────────────────────────

/**
 * Aggregate yesterday's sensor_windows rows and write summary rows for the
 * current window.  Called once per new calendar day.
 *
 * Strategy per node:
 *   - active_ratio (continuous ratio): mean of raw_value, re-discretize at cold threshold
 *   - time_of_day (label-only): majority vote on discretized_value
 *   - Default fallback: mean of raw_value when numeric, else majority vote
 */
function computePrevDayAggregates(
  db: DB,
  yesterday: string,
  todayDate: string,
  snapshotTime: string,
  windowStart: string,
): SensorWindowRow[] {
  interface HistRow {
    node_name:         string;
    source_column:     string;
    data_source:       string | null;
    raw_value:         number | null;
    raw_unit:          string | null;
    discretized_value: string;
    confidence:        number;
  }

  const rows = db.executeSync(
    `SELECT node_name, source_column, data_source, raw_value, raw_unit,
            discretized_value, confidence
     FROM   sensor_windows
     WHERE  date = ?
       AND  node_name NOT LIKE 'prev_day_%'`,
    [yesterday],
  ).rows as unknown as HistRow[];

  if (rows.length === 0) return [];

  // Group by (node_name, source_column)
  const groups: Record<string, HistRow[]> = {};
  for (const r of rows) {
    const key = `${r.node_name}||${r.source_column}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(r);
  }

  const result: SensorWindowRow[] = [];

  for (const [key, group] of Object.entries(groups)) {
    const [nodeName, sourceColumn] = key.split('||');
    const firstRow = group[0];

    // Compute aggregate raw_value (mean of non-null values)
    const numericValues = group
      .map(r => r.raw_value)
      .filter((v): v is number => v !== null && v !== undefined);
    const meanRaw = numericValues.length > 0
      ? numericValues.reduce((s, v) => s + v, 0) / numericValues.length
      : null;

    // Majority vote for discretized_value
    const voteCounts: Record<string, number> = {};
    for (const r of group) {
      voteCounts[r.discretized_value] = (voteCounts[r.discretized_value] ?? 0) + 1;
    }
    const majorityLabel = Object.entries(voteCounts).sort((a, b) => b[1] - a[1])[0][0];

    // Re-discretize for numeric nodes using the same cold-start threshold
    let discretized_value: string;
    if (nodeName === 'activity' && meanRaw !== null) {
      const activeRatio = Math.min(meanRaw / 1500, 1.0);
      discretized_value = activeRatio <= 0.25 ? 'low' : 'high';
    } else {
      // Label-only nodes (time_of_day, etc.) or no numeric value
      discretized_value = majorityLabel;
    }

    // Mean confidence
    const meanConfidence =
      group.reduce((s, r) => s + Number(r.confidence ?? 0), 0) / group.length;

    result.push({
      date:              todayDate,
      snapshot_time:     snapshotTime,
      window_start:      windowStart,
      node_name:         nodeName,
      source_column:     `prev_day_${sourceColumn}`,
      data_source:       firstRow.data_source ?? null,
      raw_value:         meanRaw,
      raw_unit:          firstRow.raw_unit ?? null,
      discretized_value,
      // Fixed below current cold-start (0.65) so current window always wins in evidence layer
      confidence:        0.60,
    });

    // Also emit prev_day_steps (total daily step count) alongside prev_day_active_ratio
    if (nodeName === 'activity' && sourceColumn === 'active_ratio' && numericValues.length > 0) {
      const totalSteps = numericValues.reduce((s, v) => s + v, 0);
      const totalRatio = Math.min(totalSteps / (numericValues.length * 1500), 1.0);
      result.push({
        date:              todayDate,
        snapshot_time:     snapshotTime,
        window_start:      windowStart,
        node_name:         'activity',
        source_column:     'prev_day_steps',
        data_source:       firstRow.data_source ?? null,
        raw_value:         totalSteps,
        raw_unit:          'steps',
        discretized_value: totalRatio <= 0.25 ? 'low' : 'high',
        confidence:        0.60,
      });
    }
  }

  return result;
}

// ── Sleep hours computation ────────────────────────────────────────────────────

/**
 * Derive last night's sleep duration from screen usage data.
 * Logic: find the longest consecutive run of 15-min windows between 8pm yesterday
 * and 11am today where screen_time_window_minutes < 1 (phone essentially unused).
 * Runs once per day after 11:00 local time.
 */
function computeSleepHours(
  db:           DB,
  yesterday:    string,
  today:        string,
  snapshotTime: string,
  windowStart:  string,
): SensorWindowRow | null {
  interface WindowRow { date: string; snapshot_time: string; raw_value: number | null }

  const rows = db.executeSync(
    `SELECT date, snapshot_time, raw_value
     FROM   sensor_windows
     WHERE  source_column = 'screen_time_window_minutes'
       AND  ((date = ? AND snapshot_time >= '20:00:00')
          OR (date = ? AND snapshot_time <= '11:00:00'))
     ORDER  BY date ASC, snapshot_time ASC`,
    [yesterday, today],
  ).rows as unknown as WindowRow[];

  if (rows.length === 0) return null;

  const MAX_GAP_MS = 20 * 60_000; // tolerate one missing window between consecutive readings
  let bestCount    = 0;
  let curCount     = 0;
  let prevMs       = 0;

  for (const row of rows) {
    const ms           = new Date(`${row.date}T${row.snapshot_time}`).getTime();
    const isUnused     = (row.raw_value ?? 1) < 1;
    const isConsec     = prevMs === 0 || (ms - prevMs) <= MAX_GAP_MS;

    if (isUnused && isConsec) {
      curCount++;
    } else {
      bestCount = Math.max(bestCount, curCount);
      curCount  = isUnused ? 1 : 0;
    }
    prevMs = ms;
  }
  bestCount = Math.max(bestCount, curCount);

  if (bestCount === 0) return null;

  const sleepHours = (bestCount * SENSING_INTERVAL_MINUTES) / 60;

  // Thresholds from feature_node_config sleep_quality/sleep_hours: [0, 6, 9, 24]
  const discretized_value = sleepHours <= 6 ? 'poor' : sleepHours <= 9 ? 'fair' : 'good';

  return {
    date:              today,
    snapshot_time:     snapshotTime,
    window_start:      windowStart,
    node_name:         'sleep_quality',
    source_column:     'sleep_hours',
    data_source:       'screen_usage_proxy',
    raw_value:         sleepHours,
    raw_unit:          'hours',
    discretized_value,
    confidence:        0.75,
  };
}

// ── Core collection logic ──────────────────────────────────────────────────────

/**
 * Run all sensor collectors and write results to sensor_windows.
 * Idempotent: uses INSERT OR REPLACE — calling multiple times in the same window
 * simply overwrites the previous row for that (date, snapshot_time, node_name, source_column).
 */
export async function runPassiveSensing(db: DB): Promise<void> {
  const date          = localDate();
  const snapshotTime  = localTime();
  const windowStart   = localTimeMinutesAgo(SENSING_INTERVAL_MINUTES);

  const collectorRows: SensorWindowRow[] = [];

  // Helper: map a CollectorResult to a SensorWindowRow for this snapshot.
  const toRow = (r: CollectorResult): SensorWindowRow => ({
    date,
    snapshot_time: snapshotTime,
    window_start:  windowStart,
    node_name:         r.node_name,
    source_column:     r.source_column,
    data_source:       r.data_source,
    raw_value:         r.raw_value,
    raw_unit:          r.raw_unit,
    discretized_value: r.discretized_value,
    confidence:        r.confidence,
  });

  // ── Phase 1 collectors ──────────────────────────────────────────────────────

  // 1. Clock → time_of_day (always succeeds, returns single result)
  const clockResult: CollectorResult = collectClock();
  collectorRows.push(toRow(clockResult));

  // 2. Pedometer → activity (single result | null on permission deny)
  const activityResult = await collectActivity(db, SENSING_INTERVAL_MINUTES);
  if (activityResult) {
    collectorRows.push(toRow(activityResult));
  }

  // 3. UsageStatsManager → screen_usage (array | null on permission deny)
  const screenUsageResults = await collectScreenUsage(db, SENSING_INTERVAL_MINUTES);
  if (screenUsageResults) {
    collectorRows.push(...screenUsageResults.map(toRow));
  }

  // 4. CallLog + SMS → communication (array | null on permission deny)
  const communicationResults = await collectCommunication(db, SENSING_INTERVAL_MINUTES);
  if (communicationResults) {
    collectorRows.push(...communicationResults.map(toRow));
  }

  // ── Phase 2+ stubs (not yet implemented) ─────────────────────────────────────
  // heart_rate, sleep_physio → native modules not yet installed; will be added in Phase 2.

  // ── Prev-day aggregation (once per new calendar day) ─────────────────────────
  if (_lastPrevDayDate !== date) {
    _lastPrevDayDate = date;
    const yesterday = previousLocalDate(date);
    const prevDayRows = computePrevDayAggregates(
      db, yesterday, date, snapshotTime, windowStart,
    );
    collectorRows.push(...prevDayRows);
  }

  // ── Sleep hours computation (once per day, after 11:00 local time) ───────────
  const currentHour = new Date().getHours();
  if (currentHour >= 11 && _lastSleepDate !== date) {
    // Guard: skip if already written for today (app restart resets in-memory flag)
    const alreadyWritten = (db.executeSync(
      `SELECT 1 FROM sensor_windows
       WHERE  node_name = 'sleep_quality' AND source_column = 'sleep_hours'
         AND  date = ? LIMIT 1`,
      [date],
    ).rows.length) > 0;
    _lastSleepDate = date;
    if (!alreadyWritten) {
      const yesterday  = previousLocalDate(date);
      const sleepRow   = computeSleepHours(db, yesterday, date, snapshotTime, windowStart);
      if (sleepRow) collectorRows.push(sleepRow);
    }
  }

  // ── Write all rows ───────────────────────────────────────────────────────────
  if (collectorRows.length > 0) {
    writeSensorReadings(db, collectorRows);
  }
}

// ── Background task definition ─────────────────────────────────────────────────
// TaskManager.defineTask must be called at module load time (before any navigator
// or component mounts) so the task exists when the OS wakes the app in background.

TaskManager.defineTask(TASK_NAME, async () => {
  if (!_db) {
    // DB not yet initialised — skip this firing
    return BackgroundFetch.BackgroundFetchResult.NoData;
  }
  try {
    await runPassiveSensing(_db);
    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch {
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Register the background fetch task and run an immediate first collection.
 * Call once after initDb() at app startup.
 *
 * Re-registration is safe — expo-background-fetch no-ops if the task is already
 * registered with the same interval.
 */
export async function initPassiveSensing(db: DB): Promise<void> {
  _db = db;

  try {
    await BackgroundFetch.registerTaskAsync(TASK_NAME, {
      minimumInterval: SENSING_INTERVAL_MINUTES * 60, // seconds
      stopOnTerminate:  false, // keep running after app is killed (Android)
      startOnBoot:      true,  // restart after device reboot (Android)
    });
  } catch {
    // Registration can fail on simulators or if already registered — not fatal.
    // The immediate call below still runs.
  }

  // Run first collection synchronously with app startup so the first inference
  // window has sensor evidence even before the background task fires.
  try {
    await runPassiveSensing(db);
  } catch {
    // First-run failures are non-fatal — permissions may not be granted yet.
  }
}
