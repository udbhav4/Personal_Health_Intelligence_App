/**
 * Evidence layer — on-device port of codebase/backend/evidence.py.
 * Reads user_data_sensorless + sensor_windows from SQLite,
 * produces evidence and prior_factors for runLBP().
 */

import type { DB } from '@op-engineering/op-sqlite';

import nodeConfig from '../assets/feature-node-config.json';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DbnEvidence {
  evidence:            Record<string, string>;
  prior_factors:       Record<string, number[]>;
  node_confidences:    Record<string, number>;
  node_data_sources:   Record<string, string>;
  sensorless_snapshot: Record<string, NodeSnapshot>;
  sensor_snapshot:     Record<string, NodeSnapshot>;
}

interface NodeSnapshot {
  node_value:  string;
  confidence:  number;
  data_source: string | null;
  created_at:  string;
}

interface SensorlessRow {
  node_value:  string;
  confidence:  number;
  data_source: string | null;
  created_at:  string;
  merge_mode:  string | null;
}

// ── Staleness windows (days) — None = latent, never queried ───────────────────

export const STALENESS_DAYS: Record<string, number | null> = {
  neuroticism:            365,
  extraversion:           365,
  age:                    365,
  sex:                    365,
  education_level:        365,
  marital_status:         365,
  bmi:                    365,
  diabetes_status:        365,
  chronic_condition:      365,
  general_health:         180,
  smoking:                180,
  alcohol_use:            180,
  loneliness:             180,
  pain_level:             90,
  positive_affect:        60,
  negative_affect:        60,
  physical_health:        30,
  mental_health:          30,
  sleep_quality:          30,
  sleep_disturbances:     30,
  stress_helplessness:    30,
  stress_self_efficacy:   30,
  depression:             30,
  stress_ema:             7,
  productivity:           7,
  social_events_positive: 7,
  social_events_negative: 7,
  exercise:               7,
  mood:                   1,
  mental_stress:          1,
  physical_stress:        1,
  time_of_day:            null,
};

const CONFIDENCE_THRESHOLD = 0.85;

// ── State labels + marginal priors (loaded from feature-node-config.json) ─────

interface NodeConfigEntry {
  state_labels?: string[];
  prior?: Record<string, Record<string, number>>;
}

function loadNodeTables(): {
  stateLabels:    Record<string, string[]>;
  marginalPriors: Record<string, number[]>;
} {
  const stateLabels:    Record<string, string[]>  = {};
  const marginalPriors: Record<string, number[]>  = {};
  const nodes = (nodeConfig as any).nodes as Record<string, NodeConfigEntry>;

  for (const [name, node] of Object.entries(nodes)) {
    const states = node.state_labels ?? [];
    stateLabels[name] = states;
    if (!states.length) continue;

    const priorDict = node.prior ?? {};
    const entries   = Object.values(priorDict);
    if (entries.length > 0) {
      const acc = Array(states.length).fill(0) as number[];
      for (const dist of entries) {
        states.forEach((s, i) => { acc[i] += (dist as Record<string, number>)[s] ?? 0; });
      }
      marginalPriors[name] = acc.map(v => v / entries.length);
    } else {
      marginalPriors[name] = Array(states.length).fill(1 / states.length);
    }
  }

  return { stateLabels, marginalPriors };
}

const { stateLabels: STATE_LABELS, marginalPriors: MARGINAL_PRIORS } = loadNodeTables();

// ── Helpers ───────────────────────────────────────────────────────────────────

function cutoffTs(days: number): string {
  const d = new Date(Date.now() - days * 86_400_000);
  return d.toISOString().replace('T', ' ').slice(0, 19);
}

function fetchNodeRows(db: DB, nodeName: string): SensorlessRow[] {
  const days = STALENESS_DAYS[nodeName];
  if (days === null || days === undefined) return [];
  const cutoff = cutoffTs(days);
  return db.executeSync(
    `SELECT node_value, confidence, data_source, created_at, merge_mode
     FROM   user_data_sensorless
     WHERE  node_name  = ?
       AND  is_active  = 1
       AND  node_value IS NOT NULL
       AND  created_at >= ?
       AND  (expires_date IS NULL OR expires_date >= date('now'))
     ORDER  BY created_at DESC`,
    [nodeName, cutoff],
  ).rows as unknown as SensorlessRow[];
}

function bestBelief(rows: SensorlessRow[]): NodeSnapshot | null {
  if (!rows.length) return null;
  const r = rows[0];
  return {
    node_value:  r.node_value,
    confidence:  Number(r.confidence),
    data_source: r.data_source,
    created_at:  r.created_at,
  };
}

function toPriorVector(
  nodeName:   string,
  nodeValue:  string,
  confidence: number,
): number[] | null {
  const states   = STATE_LABELS[nodeName];
  if (!states?.length || !states.includes(nodeValue)) return null;

  const marginal = MARGINAL_PRIORS[nodeName] ?? Array(states.length).fill(1 / states.length);
  const idx      = states.indexOf(nodeValue);
  const c1       = 1 - confidence;

  return states.map((_, i) =>
    i === idx ? confidence + c1 * marginal[i] : c1 * marginal[i],
  );
}

function fetchSensorSnapshot(db: DB): Record<string, NodeSnapshot> {
  // Step 1: find the latest (date, snapshot_time) pair
  const latest = db.executeSync(
    `SELECT date, snapshot_time
     FROM   sensor_windows
     ORDER  BY date DESC, snapshot_time DESC
     LIMIT  1`,
  ).rows[0] as { date: string; snapshot_time: string } | undefined;
  if (!latest) return {};

  // Step 2: fetch all rows for that window
  interface SensorWindowRaw {
    node_name:         string;
    source_column:     string;
    data_source:       string | null;
    discretized_value: string;
    confidence:        number;
    created_at:        string;
  }
  const rows = db.executeSync(
    `SELECT node_name, source_column, data_source, discretized_value, confidence, created_at
     FROM   sensor_windows
     WHERE  date          = ?
       AND  snapshot_time = ?`,
    [latest.date, latest.snapshot_time],
  ).rows as unknown as SensorWindowRaw[];

  // Step 3: group by node_name — keep highest-confidence row per node
  const best: Record<string, NodeSnapshot> = {};
  for (const r of rows) {
    const cf = Number(r.confidence ?? 0);
    const existing = best[r.node_name];
    if (!existing || cf > existing.confidence) {
      best[r.node_name] = {
        node_value:  r.discretized_value,
        confidence:  cf,
        data_source: r.data_source,
        created_at:  r.created_at,
      };
    }
  }
  return best;
}

/**
 * Fetch the most recent prev_day_* rows from sensor_windows (written by
 * computePrevDayAggregates). Uses the latest date that has prev_day_* entries.
 * Staleness window: 1 day (prev_day values are only meaningful for yesterday).
 */
function fetchPrevDaySnapshot(db: DB): Record<string, NodeSnapshot> {
  const todayLocal = new Date().toLocaleDateString('sv');
  interface PrevDayRaw {
    node_name:         string;
    source_column:     string;
    data_source:       string | null;
    discretized_value: string;
    confidence:        number;
    created_at:        string;
  }
  const rows = db.executeSync(
    `SELECT node_name, source_column, data_source, discretized_value, confidence, created_at
     FROM   sensor_windows
     WHERE  date = ?
       AND  source_column LIKE 'prev_day_%'`,
    [todayLocal],
  ).rows as unknown as PrevDayRaw[];

  const best: Record<string, NodeSnapshot> = {};
  for (const r of rows) {
    const cf = Number(r.confidence ?? 0);
    const existing = best[r.node_name];
    if (!existing || cf > existing.confidence) {
      best[r.node_name] = {
        node_value:  r.discretized_value,
        confidence:  cf,
        data_source: r.data_source,
        created_at:  r.created_at,
      };
    }
  }
  return best;
}

// ── Public API ────────────────────────────────────────────────────────────────

export function buildDbnEvidence(db: DB): DbnEvidence {
  const evidence:            Record<string, string>         = {};
  const prior_factors:       Record<string, number[]>       = {};
  const node_confidences:    Record<string, number>         = {};
  const node_data_sources:   Record<string, string>         = {};
  const sensorless_snapshot: Record<string, NodeSnapshot>   = {};

  // Sensorless nodes
  for (const nodeName of Object.keys(STALENESS_DAYS)) {
    const rows   = fetchNodeRows(db, nodeName);
    const belief = bestBelief(rows);
    if (!belief) continue;

    const { node_value, confidence, data_source, created_at } = belief;
    node_confidences[nodeName]  = confidence;
    node_data_sources[nodeName] = data_source ?? 'sensorless';
    sensorless_snapshot[nodeName] = { node_value, confidence, data_source, created_at };

    if (confidence >= CONFIDENCE_THRESHOLD) {
      evidence[nodeName] = node_value;
    } else {
      const vec = toPriorVector(nodeName, node_value, confidence);
      if (vec) prior_factors[nodeName] = vec;
    }
  }

  // Sensor nodes (current window)
  const sensor_snapshot = fetchSensorSnapshot(db);
  const knownSensorNodes = new Set<string>();
  for (const [nodeName, snap] of Object.entries(sensor_snapshot)) {
    const nv = snap.node_value;
    const cf = Number(snap.confidence ?? 0);
    const ds = snap.data_source ?? 'sensor';
    if (!nv) continue;

    knownSensorNodes.add(nodeName);
    node_confidences[nodeName]  = cf;
    node_data_sources[nodeName] = ds;

    // Sensorless takes precedence when both present
    if (nodeName in evidence || nodeName in prior_factors) continue;

    if (cf >= CONFIDENCE_THRESHOLD) {
      evidence[nodeName] = nv;
    } else {
      const vec = toPriorVector(nodeName, nv, cf);
      if (vec) prior_factors[nodeName] = vec;
    }
  }

  // prev_day_* nodes (written to sensor_windows by computePrevDayAggregates)
  const prevDaySnapshot = fetchPrevDaySnapshot(db);
  for (const [nodeName, snap] of Object.entries(prevDaySnapshot)) {
    const nv = snap.node_value;
    const cf = Number(snap.confidence ?? 0);
    if (!nv) continue;

    if (!(nodeName in node_confidences)) node_confidences[nodeName] = cf;
    if (!(nodeName in node_data_sources)) node_data_sources[nodeName] = snap.data_source ?? 'prev_day';

    if (nodeName in evidence || nodeName in prior_factors) continue;

    if (cf >= CONFIDENCE_THRESHOLD) {
      evidence[nodeName] = nv;
    } else {
      const vec = toPriorVector(nodeName, nv, cf);
      if (vec) prior_factors[nodeName] = vec;
    }
  }

  // Universal fallback: any DBN node still absent from both evidence and
  // prior_factors gets its trained marginal prior injected, guaranteeing cold-start
  // inference uses trained priors rather than LBP's implicit uniform assumption.
  for (const [nodeName, marginal] of Object.entries(MARGINAL_PRIORS)) {
    if (nodeName in evidence || nodeName in prior_factors) continue;
    if (!marginal?.length) continue;
    prior_factors[nodeName] = marginal;
  }

  return {
    evidence,
    prior_factors,
    node_confidences,
    node_data_sources,
    sensorless_snapshot,
    sensor_snapshot,
  };
}
