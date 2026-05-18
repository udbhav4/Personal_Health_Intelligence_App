/**
 * SQLite schema initialisation — on-device port of codebase/backend/database.py.
 *
 * Five tables:
 *   user_data_sensorless  — all user-answered questions and self-reports
 *   sensor_windows        — passive sensor evidence per inference window
 *   inference_snapshots   — full DBN output per inference run (reproducible)
 *   chat_messages         — conversation history (user + model turns)
 *   memory_summaries      — compressed long-term memory with embeddings
 *
 * sqlite-vec is auto-loaded at the native layer via the "op-sqlite" package.json config
 * ({ "sqliteVec": true }).  No JS-side loadExtension() call is needed.
 *
 * Usage:
 *   import { openDb, initDb } from './db';
 *   const db = openDb();   // call once; reuse the handle everywhere
 *   initDb(db);            // call once at app startup
 */

import { open, type DB } from '@op-engineering/op-sqlite';

const DB_NAME = 'medapp.db';

// Identical to _SCHEMA in database.py — every column, constraint, and comment preserved.
const SCHEMA = `
CREATE TABLE IF NOT EXISTS user_data_sensorless (
    -- Row identity
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                  TEXT NOT NULL,

    -- Node mapping (three-level NER routing)
    node_name                  TEXT,   -- DBN node this entry contributes to (e.g. 'depression')
    original_column            TEXT,   -- L1 NER match: item/base column (e.g. 'phq_psychomotor', 'DPQ080'), null for L2/L3
    source_column              TEXT,   -- L2 NER match: composite/direct source column (e.g. 'phq_total'), null for L1/L3

    -- Question content
    question_text              TEXT,   -- exact question shown (for audit + re-rendering)
    raw_text                   TEXT,   -- user's raw free-text or selected option label

    -- Discretized value (what the DBN consumes)
    node_value                 TEXT,   -- final discretized state string (e.g. 'mild', 'high')
    raw_value                  REAL,   -- numeric value before discretization (e.g. 2.0 for PHQ item)
    summary_text               TEXT,   -- human-readable summary shown in UI recap

    -- Evidence quality
    confidence                 REAL,   -- [0,1] certainty of node_value, fraction of scale items answered for multi-item nodes
    data_source                TEXT,   -- origin: 'self_report' | 'proactive' | 'onboarding'
    merge_mode                 TEXT,   -- how to combine multiple rows for same node: 'latest' | 'vote' | 'scale'

    -- Temporal handling
    temporal_flag              TEXT,   -- 'persistent' (trait) | 'decaying' (state)
    report_date                TEXT,   -- date the entry refers to (may differ from created_at for retrospective reports)
    expires_date               TEXT,   -- explicit expiry override, NULL = use STALENESS_DAYS from evidenceLayer

    -- Row lifecycle
    turn_id                    TEXT,                       -- UUID grouping all writes from one conversational turn (for rollback)
    is_active                  INTEGER DEFAULT 1,          -- 0 = soft-deleted / superseded
    was_proactive              INTEGER DEFAULT 0,          -- 1 = system-initiated question
    answered                   INTEGER DEFAULT 1,          -- 0 = question shown but skipped
    proactive_suppressed_until TEXT,                       -- ISO timestamp, system won't re-ask until after this

    created_at                 TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sensor_windows (
    -- Columnar schema: one row per node per inference window
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,        -- local YYYY-MM-DD
    snapshot_time     TEXT NOT NULL,        -- local YYYY-MM-DDTHH:MM:SS (window end)
    window_start      TEXT,                 -- local HH:MM:SS (window start)
    node_name         TEXT NOT NULL,        -- e.g. 'activity', 'time_of_day'
    source_column     TEXT NOT NULL,        -- e.g. 'active_ratio', 'hourly_steps', 'time_of_day'
    data_source       TEXT,                 -- 'pedometer', 'system_clock', 'healthkit', etc.
    raw_value         REAL,                 -- raw sensor reading (steps count, ratio, NULL for label-only)
    raw_unit          TEXT,                 -- 'steps', 'ratio', 'label', 'bpm'
    discretized_value TEXT,                 -- state label: 'low', 'high', 'morning', etc.
    confidence        REAL,                 -- 0.0–1.0
    created_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (date, snapshot_time, node_name, source_column)
);

CREATE TABLE IF NOT EXISTS inference_snapshots (
    -- Composite PK: one row per inference run
    date                TEXT NOT NULL,
    snapshot_time       TEXT NOT NULL,

    -- Trigger context
    trigger_type        TEXT,   -- 'scheduled' | 'sensor_event' | 'user_query'

    -- Inputs (stored for full reproducibility)
    prior_beliefs       TEXT,   -- JSON: {node: [p0, p1, ...]} — inter-slice temporal priors from t-1
    sensor_snapshot     TEXT,   -- JSON: {node: {node_value, confidence, data_source, created_at}}
    sensorless_snapshot TEXT,   -- JSON: {node: {node_value, confidence, data_source, created_at}}

    -- Outputs
    dbn_beliefs         TEXT,   -- JSON: {node: {state: prob, ...}} — full posterior from LBP
    node_confidences    TEXT,   -- JSON: {node: float}
    node_data_sources   TEXT,   -- JSON: {node: str}
    summary_line        TEXT,   -- one-sentence natural-language summary of this result

    created_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (date, snapshot_time)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    session_id  TEXT NOT NULL,              -- groups messages within one app session
    turn_id     TEXT NOT NULL,              -- UUID per conversational turn, used for rollback
    role        TEXT NOT NULL,              -- 'user' | 'model'
    content     TEXT NOT NULL,             -- raw text of the message
    topic       TEXT,                       -- NER-derived topic keywords for this turn (user rows only)
    evicted     INTEGER DEFAULT 0,          -- 1 = compressed into memory_summaries, excluded from recent-pairs buffer
    is_active   INTEGER DEFAULT 1,         -- 0 = soft-deleted (undo)
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,          -- when summary was created
    session_id      TEXT NOT NULL,
    summary_text    TEXT NOT NULL,          -- Gemma-generated compression of oldest 10 messages
    embedding       BLOB,                   -- Float32Array as raw bytes, vec_distance_cosine() operates on this BLOB
    message_count   INTEGER NOT NULL,       -- number of messages compressed into this summary
    created_at      TEXT DEFAULT (datetime('now'))
);
`;

// Singleton handle — open once, reuse everywhere (mirrors get_db() pattern in Python).
let _db: DB | null = null;

/** Open (or return cached) DB connection. Call once at app startup. */
export function openDb(): DB {
  if (!_db) _db = open({ name: DB_NAME });
  return _db;
}

/**
 * Create all tables if they don't exist.
 * Safe to call repeatedly (IF NOT EXISTS guards).
 * Mirrors init_db() in database.py.
 */
export function initDb(db: DB): void {
  probeSqliteVec(db);
  SCHEMA.split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .forEach(stmt => db.executeSync(stmt));
  runMigrations(db);
}

/**
 * Verify sqlite-vec extension is loaded by running a minimal probe query.
 * Throws a descriptive error early if the extension is missing, rather than
 * letting vec_distance_cosine fail opaquely on the first searchMemory() call.
 */
function probeSqliteVec(db: DB): void {
  try {
    db.executeSync(
      `SELECT vec_distance_cosine(X'0000803F', X'0000803F') AS probe`,
    );
  } catch (e) {
    throw new Error(
      'sqlite-vec extension not loaded — vec_distance_cosine unavailable. ' +
      'Ensure op-sqlite is built with { "sqliteVec": true } in package.json. ' +
      `Original error: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}

/**
 * Add columns introduced after the initial schema without dropping existing data.
 * Each ALTER TABLE is wrapped in try/catch — safe to run on a DB that already has
 * the column (SQLite raises "duplicate column name" which we intentionally swallow).
 *
 * Migration M002: sensor_windows was redesigned from a single-JSON-blob schema to a
 * fully columnar one.  The table was never written to in production, so DROP + CREATE
 * is safe.  We detect the old schema by checking for the now-removed sensor_data column.
 */
export function runMigrations(db: DB): void {
  // M001 — chat_messages extra columns
  const m001 = [
    `ALTER TABLE chat_messages ADD COLUMN topic   TEXT`,
    `ALTER TABLE chat_messages ADD COLUMN evicted INTEGER DEFAULT 0`,
  ];
  for (const sql of m001) {
    try { db.executeSync(sql); } catch { /* column already exists */ }
  }

  // M002 — sensor_windows: migrate old JSON-blob schema → columnar schema
  // Detect old schema by querying PRAGMA table_info and looking for the sensor_data column.
  try {
    const cols = db.executeSync(`PRAGMA table_info(sensor_windows)`).rows as unknown as Array<{ name: string }>;
    const hasOldCol = cols.some(c => c.name === 'sensor_data');
    if (hasOldCol) {
      db.executeSync(`DROP TABLE IF EXISTS sensor_windows`);
      db.executeSync(`
        CREATE TABLE IF NOT EXISTS sensor_windows (
          id                INTEGER PRIMARY KEY AUTOINCREMENT,
          date              TEXT NOT NULL,
          snapshot_time     TEXT NOT NULL,
          window_start      TEXT,
          node_name         TEXT NOT NULL,
          source_column     TEXT NOT NULL,
          data_source       TEXT,
          raw_value         REAL,
          raw_unit          TEXT,
          discretized_value TEXT,
          confidence        REAL,
          created_at        TEXT DEFAULT (datetime('now')),
          UNIQUE (date, snapshot_time, node_name, source_column)
        )
      `);
    }
  } catch { /* table may not exist yet — SCHEMA CREATE above handles that */ }

  // M003 — doctor_reports table
  try {
    db.executeSync(`CREATE TABLE IF NOT EXISTS doctor_reports (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      symptom      TEXT NOT NULL,
      file_uri     TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      created_at   TEXT DEFAULT (datetime('now'))
    )`);
  } catch { /* already exists */ }
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SensorWindowRow {
  date:              string;
  snapshot_time:     string;
  window_start?:     string | null;
  node_name:         string;
  source_column:     string;
  data_source?:      string | null;
  raw_value?:        number | null;
  raw_unit?:         string | null;
  discretized_value: string;
  confidence:        number;
}

/**
 * Upsert one or more sensor rows into sensor_windows.
 * Uses INSERT OR REPLACE so repeated calls (idempotent background task) are safe.
 */
export function writeSensorReadings(db: DB, rows: SensorWindowRow[]): void {
  for (const r of rows) {
    db.executeSync(
      `INSERT OR REPLACE INTO sensor_windows
         (date, snapshot_time, window_start, node_name, source_column,
          data_source, raw_value, raw_unit, discretized_value, confidence)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        r.date,
        r.snapshot_time,
        r.window_start     ?? null,
        r.node_name,
        r.source_column,
        r.data_source      ?? null,
        r.raw_value        ?? null,
        r.raw_unit         ?? null,
        r.discretized_value,
        r.confidence,
      ],
    );
  }
}

// ── Doctor reports ────────────────────────────────────────────────────────────

export interface DoctorReport {
  id:           number;
  symptom:      string;
  file_uri:     string;
  generated_at: string;
  created_at:   string;
}

export function writeDoctorReport(
  db: DB, symptom: string, fileUri: string, generatedAt: string,
): void {
  db.executeSync(
    `INSERT INTO doctor_reports (symptom, file_uri, generated_at) VALUES (?, ?, ?)`,
    [symptom, fileUri, generatedAt],
  );
}

export function getDoctorReports(db: DB): DoctorReport[] {
  return db.executeSync(
    `SELECT id, symptom, file_uri, generated_at, created_at
     FROM doctor_reports ORDER BY created_at DESC`,
  ).rows as unknown as DoctorReport[];
}
