/**
 * Question Cascade — post-NER guided interview.
 *
 * T1 only: siblings sharing the same source_col as a filled L1 original_col.
 * Fires only when at least one L1 entity was extracted this turn (caller-gated).
 *
 * Suppression: col is suppressed when its is_active=1 row has
 * proactive_suppressed_until > now. Caller sets this field to
 * now + STALENESS_DAYS[node] when writing a cascade answer.
 */

import type { DB } from '@op-engineering/op-sqlite';

import type { NluEntity }  from './nlu';
import columnQuestionMap   from '../assets/column-question-map.json';

export interface CascadeQuestion {
  original_col: string;
  source_col:   string;
  node_name:    string;
  question:     string;
  opts?:        { v: number; l: string }[];
  range?:       { min: number; max: number; unit: string };
  tier:         1;
}

export interface FollowUpQuestion {
  original_col: string;
  source_col:   string;
  node_name:    string;
  question:     string;
  opts?:        { v: number; l: string }[];
  range?:       { min: number; max: number; unit: string };
}

export interface CascadeState {
  questions: CascadeQuestion[];
  current:   number;
  stopped:   boolean;
}

interface ColumnEntry {
  c:         string;
  n:         string;
  q:         string;        // original question — used in NER prompt and audit
  display_q?: string;       // empathetic display text shown to user (falls back to q)
  opts?:     { v: number; l: string }[];
  range?:    { min: number; max: number; unit: string };
}

const _cqMap = columnQuestionMap as Record<string, ColumnEntry>;

function isSuppressed(db: DB, originalCol: string): boolean {
  const row = db.executeSync(
    `SELECT proactive_suppressed_until
     FROM user_data_sensorless
     WHERE original_column = ? AND is_active = 1
     ORDER BY created_at DESC
     LIMIT 1`,
    [originalCol],
  ).rows[0] as { proactive_suppressed_until: string | null } | undefined;
  if (!row?.proactive_suppressed_until) return false;
  return new Date(row.proactive_suppressed_until) > new Date();
}

function allOrigColsForSource(sourceCol: string): string[] {
  return Object.keys(_cqMap).filter(k => _cqMap[k].c === sourceCol);
}


/**
 * For L1 entities where raw_value is absent (vague phrasing, no numeric match),
 * return ordered re-ask questions so the UI can present opts/range for the user
 * to pick an explicit value. Pure lookup — no DB access.
 */
export function buildFollowUps(entities: NluEntity[]): FollowUpQuestion[] {
  const result: FollowUpQuestion[] = [];
  const seen = new Set<string>();

  for (const ent of entities) {
    if (!ent.original_column || ent.raw_value !== undefined) continue;
    if (seen.has(ent.original_column)) continue;
    const entry = _cqMap[ent.original_column];
    if (!entry) continue;
    seen.add(ent.original_column);
    const q: FollowUpQuestion = {
      original_col: ent.original_column,
      source_col:   entry.c,
      node_name:    entry.n,
      question:     entry.display_q ?? entry.q,
    };
    if (entry.opts)  q.opts  = entry.opts;
    if (entry.range) q.range = entry.range;
    result.push(q);
  }

  return result;
}

/**
 * Build cascade queue from original_cols filled by NER this pass.
 * filledCols — direct + inferred cols already written to DB.
 */
export function buildCascade(filledCols: string[], db: DB): CascadeState {
  const filled = new Set(filledCols);
  const queued = new Set<string>();
  const result: CascadeQuestion[] = [];

  // T1 only: siblings sharing the same source_col as a filled L1 original_col.
  // T2 (node-level expansion) and T3 (proactive) removed — cascade fires only
  // when L1 matched, and only asks about directly related columns.
  function tryAdd(orig: string): void {
    if (filled.has(orig) || queued.has(orig)) return;
    const entry = _cqMap[orig];
    if (!entry) return;
    if (isSuppressed(db, orig)) return;
    queued.add(orig);
    const q: CascadeQuestion = {
      original_col: orig,
      source_col:   entry.c,
      node_name:    entry.n,
      question:     entry.display_q ?? entry.q,
      tier:         1,
    };
    if (entry.opts)  q.opts  = entry.opts;
    if (entry.range) q.range = entry.range;
    result.push(q);
  }

  for (const col of filledCols) {
    const entry = _cqMap[col];
    if (!entry) continue;
    for (const sib of allOrigColsForSource(entry.c)) tryAdd(sib);
  }

  return { questions: result, current: 0, stopped: false };
}

export function currentQuestion(state: CascadeState): CascadeQuestion | null {
  if (state.stopped || state.current >= state.questions.length) return null;
  return state.questions[state.current];
}

export function advance(state: CascadeState): CascadeState {
  return { ...state, current: state.current + 1 };
}

export function stop(state: CascadeState): CascadeState {
  return { ...state, stopped: true };
}

export function isDone(state: CascadeState): boolean {
  return state.stopped || state.current >= state.questions.length;
}
