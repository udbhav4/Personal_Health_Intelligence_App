/**
 * core/reportDataCollector.ts — Doctor Report data collection layer.
 *
 * Gathers passive health evidence over a 180-day window (sensorless self-reports,
 * sensor weekly aggregates, chat excerpts, forgotten complaints, memory summaries,
 * DBN belief trajectory) and surfaces hidden patterns that the patient likely did
 * not mention during recent conversation.
 *
 * Patterns are tiered against a symptom-domain map so the report can prioritise
 * what is relevant to the chief complaint without discarding other findings.
 */

import type { DB } from '@op-engineering/op-sqlite';
import type { BeliefResult } from './inferenceEngine';
import type { UserProfile } from './AppContext';
import { embedText } from './embed';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DbnSnapshotEntry {
  created_at:        string;
  dbn_beliefs:       Record<string, Record<string, number>>;
  node_confidences:  Record<string, number>;
  node_data_sources: Record<string, string>;
  summary_line:      string | null;
}

export interface SensorlessNodeSummary {
  node_name:    string;
  node_value:   string;
  confidence:   number;
  data_source:  string | null;
  raw_text:     string | null;
  report_date:  string | null;
}

export interface SensorTrendEntry {
  week:          string;        // YYYY-WW
  node_name:     string;
  source_column: string;
  weekly_avg:    number;
  day_count:     number;
}

export type PatternType =
  | 'temporal_correlation'
  | 'anomaly_week'
  | 'sustained_trend'
  | 'forgotten_complaint'
  | 'silent_node'
  | 'contradictory_state';

export type PatternSeverity = 'notable' | 'significant' | 'critical';

export interface HiddenPattern {
  type:        PatternType;
  description: string;
  severity:    PatternSeverity;
  nodes:       string[];       // node names this pattern touches (for tier classification)
  weeks?:      string[];       // YYYY-WW labels for anomaly weeks
}

export interface ChatExcerpt {
  content:    string;
  topic:      string | null;
  created_at: string;
}

export interface MemoryRow {
  summary_text: string;
  created_at:   string;
}

export interface ReportDataObject {
  symptom:               string;
  profile:               UserProfile;
  beliefs:               BeliefResult | null;
  sensorlessSummaries:   SensorlessNodeSummary[];
  sensorTrends:          SensorTrendEntry[];
  trendDirection:        Record<string, 'up' | 'down' | 'flat'>;
  chatExcerpts:          ChatExcerpt[];
  forgottenComplaints:   ChatExcerpt[];
  memorySummaries:       MemoryRow[];
  dbnTrajectory:         DbnSnapshotEntry[];
  snapshotLine:          string | null;
  anomalyWeeks:          string[];
  silentNodes:           string[];
  keywordMatchedNodes:   Set<string>;
  tier1Patterns:         HiddenPattern[];
  tier2Patterns:         HiddenPattern[];
}

// ── Constants ─────────────────────────────────────────────────────────────────

const HIGH_RISK_DOMINANT: Record<string, string[]> = {
  depression:       ['moderate', 'moderate_severe', 'severe'],
  loneliness:       ['high'],
  pain_level:       ['significant'],
  stress_ema:       ['high'],
  mental_stress:    ['high'],
  negative_affect:  ['high', 'moderate_high'],
};

const SYMPTOM_DOMAIN_MAP: Record<string, string[]> = {
  pain:       ['pain_level','physical_stress','physical_health','exercise','bmi'],
  back:       ['pain_level','physical_stress','physical_health','exercise'],
  knee:       ['pain_level','physical_stress','exercise','bmi'],
  neck:       ['pain_level','physical_stress','stress_ema'],
  headache:   ['pain_level','mental_stress','sleep_quality','physical_stress'],
  eye:        ['pain_level','screen_time','sleep_quality','physical_stress'],
  chest:      ['pain_level','physical_stress','mental_stress','stress_ema'],
  stress:     ['mental_stress','stress_ema','stress_helplessness','stress_self_efficacy','mood','sleep_quality','productivity'],
  anxious:    ['mental_stress','stress_ema','negative_affect','sleep_quality','mood'],
  anxiety:    ['mental_stress','stress_ema','negative_affect','sleep_quality','mood'],
  depress:    ['depression','mood','negative_affect','loneliness','social_events_negative','positive_affect'],
  sad:        ['depression','mood','negative_affect','loneliness'],
  lonely:     ['loneliness','social_events_negative','mood','depression'],
  frustrated: ['mental_stress','stress_ema','mood','negative_affect','productivity'],
  angry:      ['mental_stress','negative_affect','mood'],
  overwhelm:  ['mental_stress','stress_helplessness','productivity','stress_ema'],
  sleep:      ['sleep_quality','sleep_disturbances','mental_stress','screen_time','physical_stress'],
  tired:      ['sleep_quality','physical_stress','mental_stress','bmi','physical_health'],
  fatigue:    ['sleep_quality','physical_stress','bmi','physical_health','depression'],
  exhausted:  ['sleep_quality','physical_stress','mental_stress','depression'],
  insomnia:   ['sleep_quality','sleep_disturbances','mental_stress','screen_time'],
  exercise:   ['exercise','physical_health','physical_stress','bmi'],
  active:     ['exercise','physical_health','bmi'],
  weight:     ['bmi','exercise','physical_health'],
  focus:      ['mental_stress','sleep_quality','productivity','stress_ema'],
  memory:     ['mental_stress','sleep_quality','depression'],
  productive: ['productivity','mental_stress','sleep_quality','mood'],
  work:       ['productivity','stress_ema','mental_stress','stress_helplessness'],
  social:     ['loneliness','social_events_positive','social_events_negative','mood'],
  isolated:   ['loneliness','social_events_negative','depression','mood'],
};

// "Worse" direction for each tracked node — higher = worse for these:
const HIGHER_IS_WORSE = new Set<string>([
  'stress_ema', 'mental_stress', 'physical_stress', 'depression',
  'negative_affect', 'screen_time', 'pain_level', 'sleep_disturbances',
  'stress_helplessness',
]);

const LOWER_IS_WORSE = new Set<string>([
  'mood', 'sleep_quality', 'positive_affect', 'productivity',
  'exercise', 'physical_health', 'general_health', 'stress_self_efficacy',
  'social_events_positive', 'hourly_steps', 'active_ratio',
]);

// ── Helpers ───────────────────────────────────────────────────────────────────

function safeParseJson<T>(s: string | null | undefined, fallback: T): T {
  if (!s) return fallback;
  try { return JSON.parse(s) as T; } catch { return fallback; }
}

function weeksBetween(fromIso: string, toMs: number): number {
  const fromMs = new Date(fromIso).getTime();
  if (isNaN(fromMs)) return 0;
  return Math.max(0, Math.floor((toMs - fromMs) / (7 * 86_400_000)));
}

function dominantState(dist: Record<string, number>): { state: string; prob: number } {
  let bestState = '';
  let bestProb  = -1;
  for (const [state, prob] of Object.entries(dist)) {
    if (prob > bestProb) { bestProb = prob; bestState = state; }
  }
  return { state: bestState, prob: bestProb < 0 ? 0 : bestProb };
}

function linearRegressionSlope(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX  += i;
    sumY  += values[i];
    sumXY += i * values[i];
    sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

function symptomKeywords(symptom: string): string[] {
  const s = symptom.toLowerCase();
  return Object.keys(SYMPTOM_DOMAIN_MAP).filter(k => s.includes(k));
}

function uniq<T>(arr: T[]): T[] { return Array.from(new Set(arr)); }

// Plain-English descriptions of what each DBN node means for a doctor or patient.
const NODE_DESCRIPTIONS: Record<string, string> = {
  mental_stress:        'elevated psychological stress',
  physical_stress:      'physical strain on the body',
  pain_level:           'pain',
  mood:                 'low mood',
  sleep_quality:        'poor sleep quality',
  sleep_disturbances:   'disrupted or broken sleep',
  depression:           'signs associated with depression',
  stress_ema:           'self-reported stress',
  productivity:         'reduced focus and productivity',
  exercise:             'low physical activity',
  loneliness:           'social isolation or loneliness',
  negative_affect:      'negative emotions and persistent low mood',
  positive_affect:      'reduced sense of wellbeing',
  bmi:                  'body weight concerns',
  screen_time:          'high daily screen time',
  active_ratio:         'low physical activity',
  hourly_steps:         'low daily step count',
  stress_helplessness:  'feelings of helplessness when under stress',
  stress_self_efficacy: 'reduced confidence in managing stress',
};

function nodeDesc(node: string): string {
  return NODE_DESCRIPTIONS[node] ?? node.replace(/_/g, ' ');
}


// ── Compression functions ────────────────────────────────────────────────────

/**
 * Pull up to 10 memory summaries most relevant to the stated symptom.
 * Tries cosine search first (sqlite-vec); on failure or empty result, falls back
 * to keyword match on summary_text. Each entry truncated to 120 chars.
 */
export async function compressMemories(db: DB, symptom: string): Promise<string[]> {
  const trimmed = symptom.trim();
  if (!trimmed) return [];

  let rows: Array<{ summary_text: string }> = [];

  try {
    const queryVec  = await embedText(`search_query: ${trimmed}`);
    const queryBlob = new Uint8Array(queryVec.buffer, queryVec.byteOffset, queryVec.byteLength);
    const cutoff    = new Date(Date.now() - 180 * 86_400_000)
      .toISOString().replace('T', ' ').slice(0, 19);
    rows = db.executeSync(
      `SELECT summary_text,
              vec_distance_cosine(embedding, ?) AS distance
       FROM   memory_summaries
       WHERE  embedding IS NOT NULL
         AND  created_at >= ?
       ORDER  BY distance ASC
       LIMIT  10`,
      [queryBlob, cutoff],
    ).rows as unknown as Array<{ summary_text: string }>;
  } catch {
    rows = [];
  }

  if (rows.length === 0) {
    // Keyword fallback — split symptom into tokens >2 chars
    const tokens = trimmed.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    if (tokens.length === 0) {
      rows = db.executeSync(
        `SELECT summary_text FROM memory_summaries
         WHERE created_at >= datetime('now', '-180 days')
         ORDER BY created_at DESC LIMIT 10`,
      ).rows as unknown as Array<{ summary_text: string }>;
    } else {
      const likeClauses = tokens.map(() => `summary_text LIKE ?`).join(' OR ');
      const params      = tokens.map(t => `%${t}%`);
      rows = db.executeSync(
        `SELECT summary_text FROM memory_summaries
         WHERE created_at >= datetime('now', '-180 days')
           AND (${likeClauses})
         ORDER BY created_at DESC LIMIT 10`,
        params,
      ).rows as unknown as Array<{ summary_text: string }>;
    }
  }

  return rows
    .map(r => (r.summary_text ?? '').trim())
    .filter(s => s.length > 0)
    .map(s => s.length > 120 ? s.slice(0, 117) + '...' : s);
}

/**
 * Convert a chronological list of DBN snapshots into a per-node weekly dominant
 * state string: "depression: mild,mild,moderate,moderate,moderate,severe".
 * Picks the latest snapshot per ISO-week per node.
 */
export function compressDbnTrajectory(snapshots: DbnSnapshotEntry[]): Record<string, string> {
  // node → week → latest dominant state (with ts so we can pick latest in-week)
  const byNode: Record<string, Map<string, { ts: number; state: string }>> = {};

  for (const snap of snapshots) {
    const ts = new Date(snap.created_at).getTime();
    if (isNaN(ts)) continue;
    const d  = new Date(ts);
    const wkKey = sqliteWeekStr(d);

    for (const [node, dist] of Object.entries(snap.dbn_beliefs)) {
      if (!dist) continue;
      const { state } = dominantState(dist);
      if (!state) continue;
      if (!byNode[node]) byNode[node] = new Map();
      const prev = byNode[node].get(wkKey);
      if (!prev || ts > prev.ts) byNode[node].set(wkKey, { ts, state });
    }
  }

  const result: Record<string, string> = {};
  for (const [node, weekMap] of Object.entries(byNode)) {
    const sortedWeeks = Array.from(weekMap.keys()).sort();
    const states = sortedWeeks.map(w => weekMap.get(w)!.state);
    if (states.length > 0) result[node] = states.join(',');
  }
  return result;
}

/**
 * One-line passive sensor summary comparing the first 4 vs last 4 weeks of
 * data per source_column, e.g.:
 *   "steps: 4200→2800(↓) screen: 9.1→10.3(↑)"
 */
export function compressSensorTrends(trends: SensorTrendEntry[]): string {
  if (trends.length === 0) return 'No passive sensor data available.';

  const bySource: Record<string, Array<{ week: string; avg: number }>> = {};
  for (const t of trends) {
    if (!bySource[t.source_column]) bySource[t.source_column] = [];
    bySource[t.source_column].push({ week: t.week, avg: t.weekly_avg });
  }
  for (const col of Object.keys(bySource)) {
    bySource[col].sort((a, b) => a.week.localeCompare(b.week));
  }

  const SENSOR_LABELS: Record<string, string> = {
    hourly_steps: 'steps/hr',
    active_ratio: 'activity',
    screen_time:  'screen time (min)',
  };

  const parts: string[] = [];
  for (const col of ['hourly_steps', 'active_ratio', 'screen_time']) {
    const series = bySource[col];
    if (!series || series.length < 2) continue;
    const head    = series.slice(0, 4);
    const tail    = series.slice(-4);
    const headAvg = head.reduce((s, x) => s + x.avg, 0) / head.length;
    const tailAvg = tail.reduce((s, x) => s + x.avg, 0) / tail.length;
    if (!isFinite(headAvg) || !isFinite(tailAvg)) continue;

    const fmt = (v: number) => col === 'active_ratio'
      ? v.toFixed(2)
      : Math.round(v).toString();
    const diffRatio = headAvg === 0 ? 0 : (tailAvg - headAvg) / Math.abs(headAvg);
    const dir = Math.abs(diffRatio) < 0.05 ? 'stable' : tailAvg > headAvg ? 'rising' : 'falling';
    parts.push(`${SENSOR_LABELS[col]}: ${fmt(headAvg)} → ${fmt(tailAvg)} (${dir})`);
  }
  return parts.join('; ') || 'Sensor data present but not enough weeks for a trend.';
}

/**
 * Truncate a list of HiddenPattern objects to `limit` items and each
 * description to `maxLen` characters.
 */
export function compressPatterns(
  patterns: HiddenPattern[], limit: number, maxLen: number,
): string[] {
  return patterns.slice(0, limit).map(p => {
    const d = p.description ?? '';
    return d.length > maxLen ? d.slice(0, maxLen - 3) + '...' : d;
  });
}

// ── SQLite %Y-%W week helper ──────────────────────────────────────────────────
// Matches strftime('%Y-%W', date): 4-digit year + hyphen + 2-digit week (00-53),
// where week 0 = days before the first Monday and week 1 starts on the first
// Monday of the year. We compute this in UTC so it matches the dates stored
// by SQLite's date('now', ...).

function sqliteWeekStr(date: Date): string {
  const year = date.getUTCFullYear();
  const jan1 = new Date(Date.UTC(year, 0, 1));
  const jan1Dow = jan1.getUTCDay(); // 0=Sun, 1=Mon, ..., 6=Sat
  const dayOfYear = Math.floor((date.getTime() - jan1.getTime()) / 86_400_000);
  // Offset (in days from Jan 1) of the first Monday of the year.
  const firstMondayOffset =
    jan1Dow === 0 ? 1 : jan1Dow === 1 ? 0 : 8 - jan1Dow;
  const weekNum = dayOfYear < firstMondayOffset
    ? 0
    : Math.floor((dayOfYear - firstMondayOffset) / 7) + 1;
  return `${year}-${String(weekNum).padStart(2, '0')}`;
}

// ── Pattern detection ─────────────────────────────────────────────────────────

interface WeeklyMetric {
  week:        string;
  value:       number;
  source_col:  string;
  node_name:   string;
}

/**
 * Build a per-week, per-metric value table from sensor trends plus sensorless
 * weekly snapshots (state→numeric mapping). Used by the 6 hidden-pattern
 * detectors below.
 */
function buildWeeklyMetrics(
  sensorTrends:        SensorTrendEntry[],
  sensorlessSummaries: SensorlessNodeSummary[],
  dbnTrajectory:       DbnSnapshotEntry[],
): { weeks: string[]; bySource: Record<string, Map<string, number>> } {
  const bySource: Record<string, Map<string, number>> = {};
  const weekSet  = new Set<string>();

  // From sensor trends (real numbers)
  for (const t of sensorTrends) {
    if (!bySource[t.source_column]) bySource[t.source_column] = new Map();
    bySource[t.source_column].set(t.week, t.weekly_avg);
    weekSet.add(t.week);
  }

  // From DBN trajectory — extract weekly numeric state for key nodes
  const NODES_AS_METRICS = ['stress_ema', 'mental_stress', 'mood', 'sleep_quality',
                             'depression', 'positive_affect', 'negative_affect',
                             'pain_level'];
  const stateToScore: Record<string, number> = {
    none: 0, minimal: 0.5, low: 1, mild: 1, light: 1, moderate: 2,
    moderate_severe: 2.5, mid: 2, average: 2, medium: 2,
    significant: 2.5, high: 3, severe: 3, very_high: 3.5,
    moderate_high: 2.5, very_low: 0.5,
  };

  // For each node, weekly latest snapshot mapped to a numeric score
  for (const node of NODES_AS_METRICS) {
    const weekToTs: Map<string, { ts: number; state: string }> = new Map();
    for (const snap of dbnTrajectory) {
      const ts = new Date(snap.created_at).getTime();
      if (isNaN(ts)) continue;
      const d  = new Date(ts);
      const wkKey = sqliteWeekStr(d);
      const dist = snap.dbn_beliefs[node];
      if (!dist) continue;
      const { state } = dominantState(dist);
      if (!state) continue;
      const prev = weekToTs.get(wkKey);
      if (!prev || ts > prev.ts) weekToTs.set(wkKey, { ts, state });
    }
    if (weekToTs.size === 0) continue;
    bySource[node] = new Map();
    for (const [wk, { state }] of weekToTs) {
      const score = stateToScore[state.toLowerCase()] ?? 1.5;
      bySource[node].set(wk, score);
      weekSet.add(wk);
    }
  }

  return { weeks: Array.from(weekSet).sort(), bySource };
}

/**
 * Surface six classes of hidden patterns from the gathered report data.
 * See file header comment for the per-detector rules; descriptions are short
 * enough to be cited directly by the Gemma executor calls.
 */
export function detectHiddenPatterns(data: ReportDataObject): HiddenPattern[] {
  const patterns: HiddenPattern[] = [];
  const { weeks, bySource } = buildWeeklyMetrics(
    data.sensorTrends, data.sensorlessSummaries, data.dbnTrajectory,
  );

  // ── 1) temporal_correlation ────────────────────────────────────────────────
  const sources = Object.keys(bySource);
  // weekly deltas: source -> Map(week -> sign of delta vs previous week, in 'worse' direction)
  const worseSign: Record<string, Map<string, number>> = {};
  for (const src of sources) {
    const series = weeks
      .filter(w => bySource[src].has(w))
      .map(w => ({ w, v: bySource[src].get(w)! }));
    if (series.length < 2) continue;
    const isHigherWorse = HIGHER_IS_WORSE.has(src);
    const isLowerWorse  = LOWER_IS_WORSE.has(src);
    const map: Map<string, number> = new Map();
    for (let i = 1; i < series.length; i++) {
      const delta = series[i].v - series[i - 1].v;
      if (Math.abs(delta) < 1e-6) continue;
      let sign = 0;
      if (isHigherWorse) sign = delta > 0 ? 1 : -1;
      else if (isLowerWorse) sign = delta < 0 ? 1 : -1;
      else continue;
      map.set(series[i].w, sign);
    }
    if (map.size > 0) worseSign[src] = map;
  }

  const involvedSources = Object.keys(worseSign);
  for (let i = 0; i < involvedSources.length; i++) {
    for (let j = i + 1; j < involvedSources.length; j++) {
      const a = involvedSources[i], b = involvedSources[j];
      const sharedWeeks: string[] = [];
      for (const [wk, sign] of worseSign[a]) {
        if (worseSign[b].get(wk) === sign && sign > 0) sharedWeeks.push(wk);
      }
      if (sharedWeeks.length >= 3) {
        patterns.push({
          type:        'temporal_correlation',
          severity:    'notable',
          nodes:       [a, b],
          weeks:       sharedWeeks,
          description: `${nodeDesc(a)} and ${nodeDesc(b)} moved in the same direction over ${sharedWeeks.length} weeks — they may be connected.`,
        });
      }
    }
  }
  // Promote to significant where ≥3 metrics co-worsen in the same week
  // (collected by counting how many sources share each worsening week)
  const weekCounts: Map<string, Set<string>> = new Map();
  for (const [src, m] of Object.entries(worseSign)) {
    for (const [wk, sign] of m) {
      if (sign <= 0) continue;
      if (!weekCounts.has(wk)) weekCounts.set(wk, new Set());
      weekCounts.get(wk)!.add(src);
    }
  }
  for (const [wk, srcs] of weekCounts) {
    if (srcs.size >= 3) {
      patterns.push({
        type:        'temporal_correlation',
        severity:    'significant',
        nodes:       Array.from(srcs),
        weeks:       [wk],
        description: `Several health areas — including ${Array.from(srcs).slice(0, 2).map(nodeDesc).join(' and ')} — showed concerning signs in the same week (${wk}).`,
      });
    }
  }

  // ── 2) anomaly_week ────────────────────────────────────────────────────────
  // Determine per-source tertile thresholds, then count "in worst tertile"
  // per week. ≥3 worst-tertile metrics in one week = significant, ≥5 = critical.
  const tertileBad: Record<string, Set<string>> = {}; // src -> weeks in worst tertile
  for (const [src, weekMap] of Object.entries(bySource)) {
    const values = Array.from(weekMap.values()).sort((a, b) => a - b);
    if (values.length < 4) continue;
    const tertileLow  = values[Math.floor(values.length * 1 / 3)];
    const tertileHigh = values[Math.floor(values.length * 2 / 3)];
    const worseSet = new Set<string>();
    if (HIGHER_IS_WORSE.has(src)) {
      for (const [wk, v] of weekMap) if (v >= tertileHigh) worseSet.add(wk);
    } else if (LOWER_IS_WORSE.has(src)) {
      for (const [wk, v] of weekMap) if (v <= tertileLow) worseSet.add(wk);
    }
    if (worseSet.size > 0) tertileBad[src] = worseSet;
  }
  const weekBadCount: Map<string, Set<string>> = new Map();
  for (const [src, set] of Object.entries(tertileBad)) {
    for (const wk of set) {
      if (!weekBadCount.has(wk)) weekBadCount.set(wk, new Set());
      weekBadCount.get(wk)!.add(src);
    }
  }
  for (const [wk, srcs] of weekBadCount) {
    if (srcs.size < 3) continue;
    const severity: PatternSeverity = srcs.size >= 5 ? 'critical' : 'significant';
    patterns.push({
      type:        'anomaly_week',
      severity,
      nodes:       Array.from(srcs),
      weeks:       [wk],
      description: `Week ${wk} was notably difficult — ${Array.from(srcs).slice(0, 3).map(nodeDesc).join(', ')} and others all showed poor readings at the same time.`,
    });
  }

  // ── 3) sustained_trend ─────────────────────────────────────────────────────
  // Linear slope over all weeks; if worsening for >8 consecutive weeks AND node
  // not mentioned in 14-day chat, surface as sustained_trend.
  const recentChatContent = data.chatExcerpts
    .filter(c => {
      const t = new Date(c.created_at).getTime();
      return !isNaN(t) && t > Date.now() - 14 * 86_400_000;
    })
    .map(c => (c.content || '').toLowerCase())
    .join(' ');

  for (const [src, weekMap] of Object.entries(bySource)) {
    if (weekMap.size < 9) continue;
    const orderedWeeks = Array.from(weekMap.keys()).sort();
    const values = orderedWeeks.map(w => weekMap.get(w)!);
    const slope = linearRegressionSlope(values);
    const isWorseningSlope =
      (HIGHER_IS_WORSE.has(src) && slope > 0.01) ||
      (LOWER_IS_WORSE.has(src)  && slope < -0.01);
    if (!isWorseningSlope) continue;

    // Consecutive worsening weeks (>8)
    let maxRun = 0, run = 0;
    for (let i = 1; i < values.length; i++) {
      const d = values[i] - values[i - 1];
      const isWorse =
        (HIGHER_IS_WORSE.has(src) && d > 0) ||
        (LOWER_IS_WORSE.has(src)  && d < 0);
      if (isWorse) { run++; if (run > maxRun) maxRun = run; }
      else run = 0;
    }
    if (maxRun <= 8) continue;
    if (recentChatContent.includes(src.toLowerCase())) continue;

    patterns.push({
      type:        'sustained_trend',
      severity:    'significant',
      nodes:       [src],
      description: `${nodeDesc(src)} has been gradually declining over ${maxRun + 1} weeks — this has not come up in recent conversations.`,
    });
  }

  // ── 4) forgotten_complaint ────────────────────────────────────────────────
  const nowMs = Date.now();
  for (const f of data.forgottenComplaints) {
    const wk = weeksBetween(f.created_at, nowMs);
    const excerpt = (f.content || '').slice(0, 80).replace(/\s+/g, ' ').trim();
    patterns.push({
      type:        'forgotten_complaint',
      severity:    'significant',
      nodes:       [],
      description: `[${wk} weeks ago] Patient mentioned: "${excerpt}". Not raised since.`,
    });
  }

  // ── 5) silent_node ────────────────────────────────────────────────────────
  if (data.beliefs) {
    const allChatContent = data.chatExcerpts
      .map(c => (c.content || '').toLowerCase()).join(' ');
    for (const [node, dist] of Object.entries(data.beliefs)) {
      const highRisk = HIGH_RISK_DOMINANT[node];
      if (!highRisk) continue;
      const { state, prob } = dominantState(dist);
      if (prob <= 0.6) continue;
      if (!highRisk.includes(state)) continue;
      if (allChatContent.includes(node.toLowerCase())) continue;
      patterns.push({
        type:        'silent_node',
        severity:    'significant',
        nodes:       [node],
        description: `The patient shows signs of ${nodeDesc(node)} — this has not come up in recent conversations.`,
      });
    }
  }

  // ── 6) contradictory_state ────────────────────────────────────────────────
  const vigorousExercise = data.sensorlessSummaries.find(
    s => s.node_name === 'exercise' &&
         (s.node_value === 'vigorous' || s.node_value === 'moderate') &&
         s.confidence > 0.8,
  );
  if (vigorousExercise) {
    const stepsTrend = data.sensorTrends.filter(t => t.source_column === 'hourly_steps');
    const lowSteps   = stepsTrend.filter(t => t.weekly_avg < 3000);
    if (lowSteps.length >= 3) {
      patterns.push({
        type:        'contradictory_state',
        severity:    'notable',
        nodes:       ['exercise', 'hourly_steps'],
        description: `Self-reported exercise is ${vigorousExercise.node_value}, but step counts averaged <3000/day in ${lowSteps.length} weeks.`,
      });
    }
  }

  // anomaly weeks are surfaced via patterns[].weeks for type === 'anomaly_week';
  // the caller (gatherReportData) collects them from there.
  return patterns;
}

/**
 * Split detected patterns into Tier 1 (symptom-linked / always-promoted) and
 * Tier 2 (everything else). Falls back to a simple type-based rule when the
 * symptom does not map to any node domain.
 */
export function classifyPatternTiers(
  patterns: HiddenPattern[], symptom: string,
): { tier1: HiddenPattern[]; tier2: HiddenPattern[] } {
  const keys          = symptomKeywords(symptom);
  const relevantNodes = new Set<string>();
  for (const k of keys) {
    for (const n of SYMPTOM_DOMAIN_MAP[k] ?? []) relevantNodes.add(n);
  }

  // Passive sensor nodes always relevant for any complaint
  relevantNodes.add('screen_time');
  relevantNodes.add('activity');

  const tier1: HiddenPattern[] = [];
  const tier2: HiddenPattern[] = [];

  for (const p of patterns) {
    if (relevantNodes.size === 0) {
      if (p.type === 'forgotten_complaint' || p.type === 'anomaly_week') tier1.push(p);
      else tier2.push(p);
      continue;
    }
    const symptomLinked = p.nodes.some(n => relevantNodes.has(n));
    const alwaysT1      = p.type === 'forgotten_complaint' || p.type === 'anomaly_week';
    const critical      = p.severity === 'critical';
    if (symptomLinked || alwaysT1 || critical) tier1.push(p);
    else tier2.push(p);
  }

  return { tier1, tier2 };
}

// ── Trend direction per source column ────────────────────────────────────────

function computeTrendDirection(
  trends: SensorTrendEntry[],
): Record<string, 'up' | 'down' | 'flat'> {
  const bySource: Record<string, Array<{ week: string; avg: number }>> = {};
  for (const t of trends) {
    if (!bySource[t.source_column]) bySource[t.source_column] = [];
    bySource[t.source_column].push({ week: t.week, avg: t.weekly_avg });
  }
  const result: Record<string, 'up' | 'down' | 'flat'> = {};
  for (const [col, series] of Object.entries(bySource)) {
    series.sort((a, b) => a.week.localeCompare(b.week));
    if (series.length < 2) { result[col] = 'flat'; continue; }
    const head = series.slice(0, 4).reduce((s, x) => s + x.avg, 0) / Math.min(4, series.length);
    const tail = series.slice(-4).reduce((s, x) => s + x.avg, 0) / Math.min(4, series.length);
    const diff = head === 0 ? 0 : (tail - head) / Math.abs(head);
    if (Math.abs(diff) < 0.05) result[col] = 'flat';
    else if (diff > 0) result[col] = 'up';
    else result[col] = 'down';
  }
  return result;
}

// ── Main entry point ─────────────────────────────────────────────────────────

/**
 * Gather every piece of evidence the Doctor Report needs. Runs ~6 SQL queries
 * across a 180-day window, detects hidden patterns, and tiers them against the
 * stated symptom.  Returns one self-contained object the HTML builder + Gemma
 * narrative pipeline both consume.
 */
export async function gatherReportData(
  db:      DB,
  symptom: string,
  profile: UserProfile,
  beliefs: BeliefResult | null,
): Promise<ReportDataObject> {

  // ── Query A: sensorless summaries (highest-confidence per node, >0.3) ──────
  interface SensorlessRow {
    node_name:   string;
    node_value:  string;
    confidence:  number;
    data_source: string | null;
    raw_text:    string | null;
    report_date: string | null;
  }
  let sensorlessRows: SensorlessRow[] = [];
  try {
    sensorlessRows = db.executeSync(
      `SELECT node_name, node_value, confidence, data_source, raw_text, report_date
       FROM user_data_sensorless
       WHERE is_active = 1 AND report_date >= date('now', '-180 days')
       ORDER BY node_name, report_date DESC`,
    ).rows as unknown as SensorlessRow[];
  } catch { sensorlessRows = []; }

  const bestPerNode: Map<string, SensorlessRow> = new Map();
  for (const r of sensorlessRows) {
    if (r.confidence == null || r.confidence <= 0.3) continue;
    if (!r.node_name || !r.node_value) continue;
    const prev = bestPerNode.get(r.node_name);
    if (!prev || (r.confidence ?? 0) > (prev.confidence ?? 0)) {
      bestPerNode.set(r.node_name, r);
    }
  }
  const sensorlessSummaries: SensorlessNodeSummary[] = Array.from(bestPerNode.values()).map(r => ({
    node_name:   r.node_name,
    node_value:  r.node_value,
    confidence:  r.confidence,
    data_source: r.data_source,
    raw_text:    r.raw_text,
    report_date: r.report_date,
  }));

  // ── Query B: sensor weekly aggregates ──────────────────────────────────────
  let sensorTrends: SensorTrendEntry[] = [];
  try {
    sensorTrends = db.executeSync(
      `SELECT strftime('%Y-%W', date) AS week, node_name, source_column,
              AVG(CAST(raw_value AS REAL)) AS weekly_avg, COUNT(*) AS day_count
       FROM sensor_windows
       WHERE date >= date('now', '-180 days')
         AND source_column IN ('hourly_steps', 'active_ratio', 'screen_time')
       GROUP BY week, node_name, source_column
       ORDER BY week ASC, node_name ASC`,
    ).rows as unknown as SensorTrendEntry[];
  } catch { sensorTrends = []; }

  // ── Query C: recent chat (30 user messages, >20 chars) ─────────────────────
  let chatRows: ChatExcerpt[] = [];
  try {
    chatRows = (db.executeSync(
      `SELECT content, topic, created_at FROM chat_messages
       WHERE role = 'user' AND is_active = 1 AND evicted = 0
         AND created_at >= datetime('now', '-180 days')
       ORDER BY created_at DESC LIMIT 30`,
    ).rows as unknown as ChatExcerpt[])
      .filter(r => (r.content ?? '').length > 20);
  } catch { chatRows = []; }

  const keywords = symptomKeywords(symptom);
  const relevant = chatRows.filter(r => {
    const c = (r.content ?? '').toLowerCase();
    return keywords.some(k => c.includes(k));
  });
  const chatExcerpts: ChatExcerpt[] = (relevant.length > 0 ? relevant : chatRows).slice(0, 5);

  // ── Query F: forgotten complaints ──────────────────────────────────────────
  let forgottenComplaints: ChatExcerpt[] = [];
  try {
    forgottenComplaints = db.executeSync(
      `SELECT content, topic, created_at FROM chat_messages
       WHERE role = 'user' AND is_active = 1 AND evicted = 0
         AND created_at < datetime('now', '-60 days')
         AND created_at >= datetime('now', '-180 days')
         AND (content LIKE '%pain%' OR content LIKE '%tired%' OR content LIKE '%stress%'
           OR content LIKE '%sleep%' OR content LIKE '%anxious%' OR content LIKE '%depress%'
           OR content LIKE '%hurt%' OR content LIKE '%sick%' OR content LIKE '%fatigue%'
           OR content LIKE '%ache%' OR content LIKE '%numb%' OR content LIKE '%dizzy%'
           OR content LIKE '%headache%' OR content LIKE '%back%' OR content LIKE '%eye%')
       ORDER BY created_at DESC LIMIT 15`,
    ).rows as unknown as ChatExcerpt[];
  } catch { forgottenComplaints = []; }

  // ── Query D: memory summaries ──────────────────────────────────────────────
  let memoryRows: MemoryRow[] = [];
  try {
    memoryRows = db.executeSync(
      `SELECT summary_text, created_at FROM memory_summaries
       WHERE created_at >= datetime('now', '-180 days')
       ORDER BY created_at DESC LIMIT 50`,
    ).rows as unknown as MemoryRow[];
  } catch { memoryRows = []; }

  // ── Query E: DBN trajectory (inference_snapshots) ─────────────────────────
  interface RawSnapshotRow {
    dbn_beliefs:       string | null;
    node_confidences:  string | null;
    node_data_sources: string | null;
    summary_line:      string | null;
    created_at:        string;
  }
  let snapshotRaw: RawSnapshotRow[] = [];
  try {
    snapshotRaw = db.executeSync(
      `SELECT dbn_beliefs, node_confidences, node_data_sources, summary_line, created_at
       FROM inference_snapshots WHERE created_at >= datetime('now', '-180 days')
       ORDER BY created_at ASC`,
    ).rows as unknown as RawSnapshotRow[];
  } catch { snapshotRaw = []; }

  const dbnTrajectory: DbnSnapshotEntry[] = snapshotRaw.map(r => ({
    created_at:        r.created_at,
    dbn_beliefs:       safeParseJson<Record<string, Record<string, number>>>(r.dbn_beliefs, {}),
    node_confidences:  safeParseJson<Record<string, number>>(r.node_confidences, {}),
    node_data_sources: safeParseJson<Record<string, string>>(r.node_data_sources, {}),
    summary_line:      r.summary_line,
  }));
  const snapshotLine = dbnTrajectory.length > 0
    ? dbnTrajectory[dbnTrajectory.length - 1].summary_line
    : null;

  // ── Derived: trend direction per source column ────────────────────────────
  const trendDirection = computeTrendDirection(sensorTrends);

  // ── Build a draft ReportDataObject so detectors can read everything ───────
  const draft: ReportDataObject = {
    symptom,
    profile,
    beliefs,
    sensorlessSummaries,
    sensorTrends,
    trendDirection,
    chatExcerpts,
    forgottenComplaints,
    memorySummaries:     memoryRows,
    dbnTrajectory,
    snapshotLine,
    anomalyWeeks:        [],
    silentNodes:         [],
    keywordMatchedNodes: new Set<string>(),
    tier1Patterns:       [],
    tier2Patterns:       [],
  };

  // ── Patterns ──────────────────────────────────────────────────────────────
  const patterns = detectHiddenPatterns(draft);

  // anomalyWeeks / silentNodes pulled from detected patterns
  const anomalyWeeks: string[] = [];
  const silentNodes:  string[] = [];
  for (const p of patterns) {
    if (p.type === 'anomaly_week' && p.weeks) anomalyWeeks.push(...p.weeks);
    if (p.type === 'silent_node') silentNodes.push(...p.nodes);
  }

  // keywordMatchedNodes — nodes whose name appears in any chat content
  const allChatContent = chatExcerpts
    .map(c => (c.content || '').toLowerCase()).join(' ');
  const keywordMatched = new Set<string>();
  if (beliefs) {
    for (const node of Object.keys(beliefs)) {
      if (allChatContent.includes(node.toLowerCase())) keywordMatched.add(node);
    }
  }

  const { tier1, tier2 } = classifyPatternTiers(patterns, symptom);

  return {
    ...draft,
    anomalyWeeks:        uniq(anomalyWeeks),
    silentNodes:         uniq(silentNodes),
    keywordMatchedNodes: keywordMatched,
    tier1Patterns:       tier1,
    tier2Patterns:       tier2,
  };
}
