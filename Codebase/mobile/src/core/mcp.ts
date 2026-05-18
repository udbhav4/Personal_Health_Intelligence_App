/**
 * MCP tool layer — manual protocol, no SDK.
 * All agent tools are exposed via MCP only.
 *
 * Usage:
 *   const result = await dispatchTool('run_dbn_inference', { turn_id: 'abc' }, db, 'abc');
 */

import type { DB } from '@op-engineering/op-sqlite';

import { buildDbnEvidence }                                            from './evidenceLayer';
import { runLBP, applyInterSlice, formatBeliefs, MODEL_PARENTS, MODEL_CHILDREN, MODEL_STATES }
  from './inferenceEngine';
import type { BeliefResult }                                           from './inferenceEngine';
import { buildCascade }                                                from './questionCascade';
import { searchMemory }                                                from './embed';

// ── Types ─────────────────────────────────────────────────────────────────────

interface McpTool {
  name:        string;
  description: string;
  inputSchema: { type: 'object'; properties: Record<string, unknown>; required?: string[] };
  handler:     (args: Record<string, unknown>, db: DB, turnId: string) => unknown | Promise<unknown>;
}

// ── Session state (in-memory, reset on app cold-start) ────────────────────────

interface SessionState {
  lastInferenceAt:          number;
  lastEvidenceWriteAt:      number;
  turnRunCount:             number;
  currentTurnId:            string;
  currentUserMessage:       string;
  recentTopics:             string;  // joined topics from last X user messages; enriches memory search query
  latestBeliefs:            BeliefResult | null;
  turnStartBeliefs:         BeliefResult | null;  // baseline for get_changed_nodes per turn
  turnStartOverrideActive:  boolean;              // true when overrideTurnStart() has been called this turn
  latestEvidenceNodes:      string[];
  directEvidenceNodes:      Set<string>;          // NER-extracted nodes this turn; indirect evidence must not overwrite
}

const _session: SessionState = {
  lastInferenceAt:         0,
  lastEvidenceWriteAt:     0,
  turnRunCount:            0,
  currentTurnId:           '',
  currentUserMessage:      '',
  recentTopics:            '',
  latestBeliefs:           null,
  turnStartBeliefs:        null,
  turnStartOverrideActive: false,
  latestEvidenceNodes:     [],
  directEvidenceNodes:     new Set(),
};

export function resetSession(): void {
  _session.lastInferenceAt         = 0;
  _session.lastEvidenceWriteAt     = 0;
  _session.turnRunCount            = 0;
  _session.currentTurnId           = '';
  _session.currentUserMessage      = '';
  _session.recentTopics            = '';
  _session.latestBeliefs           = null;
  _session.turnStartBeliefs        = null;
  _session.turnStartOverrideActive = false;
  _session.latestEvidenceNodes     = [];
  _session.directEvidenceNodes     = new Set();
}

/** Called by agent.ts at the start of each turn so get_user_memory can do semantic search. */
export function setCurrentUserMessage(text: string): void {
  _session.currentUserMessage = text;
}

/** Called by agent.ts with topics from recent pairs before get_user_memory fires, enriching the search query. */
export function setRecentTopics(topics: string): void {
  _session.recentTopics = topics;
}

/** Called by agent.ts after writing proactive answers so the inference throttle allows a second run. */
export function markEvidenceWritten(): void {
  _session.lastEvidenceWriteAt = Date.now();
}

/** Called by agent.ts with NER entity nodes so store_indirect_evidence cannot overwrite direct evidence. */
export function setDirectEvidenceNodes(nodes: Set<string>): void {
  _session.directEvidenceNodes = nodes;
}

/**
 * Called by agent.ts after undo_last_entry fires, before run_dbn_inference.
 * Pins the turn-start baseline to the pre-correction snapshot so
 * get_changed_nodes diffs from before the wrong data entered.
 */
export function overrideTurnStart(beliefs: BeliefResult): void {
  _session.turnStartBeliefs        = beliefs;
  _session.turnStartOverrideActive = true;
}

// ── Topology helpers ──────────────────────────────────────────────────────────

function getAncestors(node: string): Set<string> {
  const visited = new Set<string>();
  const queue   = [...(MODEL_PARENTS[node] ?? [])];
  while (queue.length) {
    const n = queue.pop()!;
    if (visited.has(n)) continue;
    visited.add(n);
    for (const p of MODEL_PARENTS[n] ?? []) queue.push(p);
  }
  return visited;
}


function getDescendants(node: string): Set<string> {
  const visited = new Set<string>();
  const queue   = [...(MODEL_CHILDREN[node] ?? [])];
  while (queue.length) {
    const n = queue.pop()!;
    if (visited.has(n)) continue;
    visited.add(n);
    for (const c of MODEL_CHILDREN[n] ?? []) queue.push(c);
  }
  return visited;
}


function argmaxState(dist: Record<string, number>): string {
  let best = ''; let bestP = -1;
  for (const [s, p] of Object.entries(dist)) { if (p > bestP) { best = s; bestP = p; } }
  return best;
}

// ── Tool handlers ─────────────────────────────────────────────────────────────

// alpha/tMax params per mode: glance (window_days=14) → 0.4/14, reflect (window_days=90) → 0.2/90
function decayParams(windowDays: number): { alpha: number; tMax: number } {
  return windowDays === 14
    ? { alpha: 0.4, tMax: 14 }
    : { alpha: 0.2, tMax: 90 };
}

async function handleGetUserMemory(args: Record<string, unknown>, db: DB): Promise<unknown> {
  const windowDays      = typeof args.window_days === 'number' ? args.window_days : 14;
  const { alpha, tMax } = decayParams(windowDays);
  // Enrich query with recent topics so older memories relevant to current context surface
  const query = [_session.recentTopics, _session.currentUserMessage].filter(Boolean).join(' ');
  const results = await searchMemory(db, query, windowDays, 5, alpha, tMax);
  return { summaries: results.map(r => r.summary_text) };
}


function handleRunDbnInference(args: Record<string, unknown>, db: DB, turnId: string): unknown {
  const tid = typeof args.turn_id === 'string' ? args.turn_id : turnId;

  // New turn: reset run counter and save baseline (skip baseline reset if undo override is active)
  if (_session.currentTurnId !== tid) {
    _session.currentTurnId = tid;
    _session.turnRunCount  = 0;
    if (!_session.turnStartOverrideActive) {
      _session.turnStartBeliefs = _session.latestBeliefs;
    }
  }

  if (_session.turnRunCount >= 2) {
    return { skipped: true, reason: 'max_runs_reached', beliefs: _session.latestBeliefs };
  }
  if (_session.turnRunCount > 0 && _session.lastEvidenceWriteAt <= _session.lastInferenceAt) {
    return { skipped: true, reason: 'no_new_evidence', beliefs: _session.latestBeliefs };
  }

  const dbnEvidence = buildDbnEvidence(db);
  const { evidence, prior_factors, node_confidences, node_data_sources, sensorless_snapshot, sensor_snapshot } = dbnEvidence;

  // Fetch previous DB snapshot for temporal priors (read BEFORE writing new one)
  const prevRow = db.executeSync(
    `SELECT dbn_beliefs FROM inference_snapshots ORDER BY date DESC, snapshot_time DESC LIMIT 1`,
  ).rows[0] as { dbn_beliefs: string } | undefined;

  const priorFactors = { ...prior_factors };
  let prevBeliefs: BeliefResult | null = null;
  if (prevRow?.dbn_beliefs) {
    try {
      prevBeliefs = JSON.parse(prevRow.dbn_beliefs) as BeliefResult;
      const temporal = applyInterSlice(prevBeliefs);
      for (const [node, vec] of Object.entries(temporal)) {
        if (!(node in priorFactors)) priorFactors[node] = vec;
      }
    } catch { /* ignore corrupt snapshot */ }
  }

  const beliefResult = formatBeliefs(runLBP(evidence, priorFactors));

  const now     = new Date();
  const dateStr = now.toLocaleDateString('sv');
  const timeStr = now.toLocaleString('sv').replace(' ', 'T').slice(0, 19);

  db.executeSync(
    `INSERT OR REPLACE INTO inference_snapshots
       (date, snapshot_time, trigger_type, prior_beliefs,
        sensor_snapshot, sensorless_snapshot, dbn_beliefs,
        node_confidences, node_data_sources)
     VALUES (?, ?, 'user_query', ?, ?, ?, ?, ?, ?)`,
    [
      dateStr, timeStr,
      JSON.stringify(priorFactors),
      JSON.stringify(sensor_snapshot),
      JSON.stringify(sensorless_snapshot),
      JSON.stringify(beliefResult),
      JSON.stringify(node_confidences),
      JSON.stringify(node_data_sources),
    ],
  );

  // On first run of first-ever turn, seed turn-start baseline from DB
  if (_session.turnStartBeliefs === null && prevBeliefs !== null) {
    _session.turnStartBeliefs = prevBeliefs;
  }

  _session.latestBeliefs       = beliefResult;
  _session.latestEvidenceNodes = Object.keys(evidence);
  _session.lastInferenceAt     = Date.now();
  _session.turnRunCount++;

  return { beliefs: beliefResult };
}


function handleGetChangedNodes(): unknown {
  _session.turnStartOverrideActive = false;  // consumed — clear so next turn gets a fresh baseline
  const { latestBeliefs, turnStartBeliefs, latestEvidenceNodes } = _session;
  if (!latestBeliefs || !turnStartBeliefs) return { changed_nodes: [] };

  const evidenceSet = new Set(latestEvidenceNodes);

  // Pre-compute ancestor/descendant sets for all evidence nodes
  const evidenceAncestors   = new Set<string>();
  const evidenceDescendants = new Set<string>();
  const evidenceParents     = new Set<string>();

  for (const ev of evidenceSet) {
    for (const a of getAncestors(ev))   evidenceAncestors.add(a);
    for (const d of getDescendants(ev)) evidenceDescendants.add(d);
    for (const p of MODEL_PARENTS[ev] ?? []) evidenceParents.add(p);
  }

  // co-influenced = shares a parent with any evidence node, but is not itself evidence
  const coInfluenced = new Set<string>();
  for (const p of evidenceParents) {
    for (const c of MODEL_CHILDREN[p] ?? []) {
      if (!evidenceSet.has(c)) coInfluenced.add(c);
    }
  }

  type ChangedNode = { node: string; label: string; previous_state: string; new_state: string; delta: number };
  const changed: ChangedNode[] = [];

  for (const [node, newDist] of Object.entries(latestBeliefs)) {
    const oldDist = turnStartBeliefs[node];
    if (!oldDist) continue;

    let delta = 0;
    for (const [state, pNew] of Object.entries(newDist)) {
      delta = Math.max(delta, Math.abs(pNew - (oldDist[state] ?? 0)));
    }
    if (delta < 0.05) continue;

    let label: string;
    if (evidenceSet.has(node))           label = 'evidence';
    else if (evidenceAncestors.has(node)) label = 'cause';
    else if (coInfluenced.has(node))      label = 'co-influenced';
    else                                  label = 'effect';

    changed.push({
      node,
      label,
      previous_state: argmaxState(oldDist),
      new_state:      argmaxState(newDist),
      delta,
    });
  }

  changed.sort((a, b) => b.delta - a.delta);
  return { changed_nodes: changed.slice(0, 6) };
}


function handleUndoLastEntry(args: Record<string, unknown>, db: DB): unknown {
  const tid = typeof args.turn_id === 'string' ? args.turn_id : '';
  if (!tid) return { undone: false, reason: 'missing_turn_id' };

  const nodeNames = Array.isArray(args.node_names)
    ? (args.node_names as unknown[]).filter((n): n is string => typeof n === 'string')
    : null;

  let r1: { rowsAffected: number };
  if (nodeNames && nodeNames.length > 0) {
    // Entity-level: delete only specified nodes from that turn
    const placeholders = nodeNames.map(() => '?').join(', ');
    r1 = db.executeSync(
      `UPDATE user_data_sensorless SET is_active = 0
       WHERE turn_id = ? AND node_name IN (${placeholders}) AND is_active = 1`,
      [tid, ...nodeNames],
    );
  } else {
    r1 = db.executeSync(
      `UPDATE user_data_sensorless SET is_active = 0 WHERE turn_id = ? AND is_active = 1`,
      [tid],
    );
  }

  const r2 = db.executeSync(
    `UPDATE chat_messages SET is_active = 0 WHERE turn_id = ? AND is_active = 1`,
    [tid],
  );

  // Fetch snapshot BEFORE the undone turn (second-most-recent) and return it.
  // Agent.ts calls overrideTurnStart(pre_undo_beliefs) after run_dbn_inference
  // to pin get_changed_nodes baseline to the pre-correction state.
  const snapRows = db.executeSync(
    `SELECT dbn_beliefs FROM inference_snapshots ORDER BY date DESC, snapshot_time DESC LIMIT 2`,
  ).rows as { dbn_beliefs: string }[];
  const preUndoSnap = snapRows[1] ?? snapRows[0] ?? null;
  let preUndoBeliefs: BeliefResult | null = null;
  if (preUndoSnap?.dbn_beliefs) {
    try { preUndoBeliefs = JSON.parse(preUndoSnap.dbn_beliefs) as BeliefResult; } catch {}
  }

  return { undone: true, sensorless_rows: r1.rowsAffected, chat_rows: r2.rowsAffected, pre_undo_beliefs: preUndoBeliefs };
}


function handleStoreIndirectEvidence(args: Record<string, unknown>, db: DB, turnId: string): unknown {
  const nodeName   = typeof args.node_name   === 'string' ? args.node_name   : null;
  const nodeValue  = typeof args.node_value  === 'string' ? args.node_value  : null;
  const confidence = typeof args.confidence  === 'number' ? args.confidence  : 0.5;
  const summary    = typeof args.summary     === 'string' ? args.summary     : '';

  if (!nodeName || !nodeValue) return { stored: false, reason: 'missing_fields' };

  const validNodes = Object.keys(MODEL_PARENTS);
  if (!validNodes.includes(nodeName)) {
    return { stored: false, reason: 'invalid_node_name' };
  }

  const validValues = MODEL_STATES[nodeName];
  if (!validValues || !validValues.includes(nodeValue)) {
    return { stored: false, reason: 'invalid_node_value', valid_values: validValues ?? [] };
  }

  if (_session.directEvidenceNodes.has(nodeName)) return { stored: false, reason: 'node_has_direct_evidence' };

  const now   = new Date();
  const ts    = now.toISOString().replace('T', ' ').slice(0, 19);
  const today = now.toISOString().slice(0, 10);

  db.executeSync(
    `INSERT INTO user_data_sensorless
       (timestamp, node_name, node_value, confidence, data_source,
        merge_mode, temporal_flag, report_date, summary_text, turn_id,
        was_proactive, answered)
     VALUES (?, ?, ?, ?, 'agent_indirect', 'latest', 'decaying', ?, ?, ?, 0, 1)`,
    [ts, nodeName, nodeValue, confidence, today, summary, turnId || null],
  );

  _session.lastEvidenceWriteAt = Date.now();
  return { stored: true };
}

function handleGetBeliefTrend(args: Record<string, unknown>, db: DB): unknown {
  const nodeNames  = Array.isArray(args.node_names)
    ? (args.node_names as unknown[]).filter((n): n is string => typeof n === 'string')
    : [];
  const windowDays = typeof args.window_days === 'number' ? args.window_days : 14;

  if (nodeNames.length === 0) return { trends: {} };

  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - windowDays);
  const cutoffStr = cutoff.toISOString().slice(0, 10);

  const trends: Record<string, string> = {};

  for (const node of nodeNames) {
    const rows = db.executeSync(
      `SELECT raw_text, raw_value, report_date
       FROM user_data_sensorless
       WHERE node_name = ? AND is_active = 1 AND report_date >= ?
       ORDER BY report_date ASC, timestamp ASC`,
      [node, cutoffStr],
    ).rows as { raw_text: string; raw_value: number | null; report_date: string }[];

    if (rows.length < 3) { trends[node] = ''; continue; }

    // Split into 3 equal segments by index (not by date)
    const third = Math.ceil(rows.length / 3);
    const segments = [
      rows.slice(0, third),
      rows.slice(third, third * 2),
      rows.slice(third * 2),
    ];
    const segLabels = ['early', 'mid', 'late'] as const;

    // Compute overall mean across all numeric rows (for direction threshold)
    const allNumeric = rows.filter(r => r.raw_value !== null) as { raw_text: string; raw_value: number; report_date: string }[];
    const overallMean = allNumeric.length > 0
      ? allNumeric.reduce((s, r) => s + r.raw_value, 0) / allNumeric.length
      : null;
    const threshold = overallMean !== null ? (overallMean * 0.1 || 0.1) : 0.1;

    // Per-segment stats
    const segStats: Array<{ mode: string; mean: number | null }> = segments.map(seg => {
      if (seg.length === 0) return { mode: '', mean: null };

      // mode of raw_text
      const freq: Record<string, number> = {};
      for (const r of seg) { freq[r.raw_text] = (freq[r.raw_text] ?? 0) + 1; }
      let segMode = seg[0].raw_text;
      let bestCount = 0;
      for (const [text, count] of Object.entries(freq)) {
        if (count > bestCount) { segMode = text; bestCount = count; }
      }

      // mean of numeric values
      const numeric = seg.filter(r => r.raw_value !== null) as { raw_text: string; raw_value: number; report_date: string }[];
      const segMean = numeric.length > 0
        ? numeric.reduce((s, r) => s + r.raw_value, 0) / numeric.length
        : null;

      return { mode: segMode, mean: segMean };
    });

    // Build segment strings
    const segStrings: string[] = segments.map((seg, i) => {
      if (seg.length === 0) return `[${segLabels[i]}: (no data)]`;

      const { mode, mean } = segStats[i];

      // Direction vs previous segment (only meaningful for mid and late)
      let dir = '';
      if (i > 0 && mean !== null && segStats[i - 1].mean !== null) {
        const prevMean = segStats[i - 1].mean as number;
        if (mean > prevMean + threshold)      dir = ' ↑';
        else if (mean < prevMean - threshold) dir = ' ↓';
        else                                  dir = ' →';
      }

      return `[${segLabels[i]}: ${mode}${mean !== null ? ' ' + mean.toFixed(1) : ''}${dir}]`;
    });

    trends[node] = `${node}: ${segStrings.join(' ')}`;
  }

  return { trends };
}


function handleGetCascadeQuestions(args: Record<string, unknown>, db: DB): unknown {
  const tid = typeof args.turn_id === 'string' ? args.turn_id : '';

  const rows = db.executeSync(
    `SELECT DISTINCT original_column
     FROM   user_data_sensorless
     WHERE  turn_id = ? AND original_column IS NOT NULL AND is_active = 1`,
    [tid],
  ).rows as { original_column: string }[];

  const state = buildCascade(rows.map(r => r.original_column), db);
  return { questions: state.questions.map(q => q.question) };
}

// ── Tool definitions ──────────────────────────────────────────────────────────

export const MCP_TOOLS: McpTool[] = [
  {
    name:        'get_user_memory',
    description: 'Retrieve recent memory summaries from the user\'s conversation history, ranked by recency and importance.',
    inputSchema: {
      type:       'object',
      properties: { window_days: { type: 'number', description: 'Look-back window in days (14 for glance, 90 for reflect).' } },
      required:   ['window_days'],
    },
    handler: (args, db) => handleGetUserMemory(args, db),
  },
  {
    name:        'run_dbn_inference',
    description: 'Run Loopy Belief Propagation on current evidence. Throttled to 2 runs per turn; skips if no new evidence since last run.',
    inputSchema: {
      type:       'object',
      properties: { turn_id: { type: 'string', description: 'Current turn UUID.' } },
      required:   ['turn_id'],
    },
    handler: (args, db, tid) => handleRunDbnInference(args, db, tid),
  },
  {
    name:        'get_changed_nodes',
    description: 'Compare latest inference vs turn-start beliefs. Returns up to 6 nodes with delta ≥ 0.05, labelled cause/effect.',
    inputSchema: { type: 'object', properties: {} },
    handler: () => handleGetChangedNodes(),
  },
  {
    name:        'undo_last_entry',
    description: 'Soft-delete all user_data_sensorless and chat_message rows for a given turn_id.',
    inputSchema: {
      type:       'object',
      properties: {
        turn_id:    { type: 'string', description: 'Turn UUID to undo.' },
        node_names: { type: 'array', items: { type: 'string' }, description: 'Nodes to undo (entity-level). Omit for full turn undo.' },
      },
      required: ['turn_id'],
    },
    handler: (args, db) => handleUndoLastEntry(args, db),
  },
  {
    name:        'store_indirect_evidence',
    description: 'Write agent-reasoned (L3-style) evidence for a DBN node. Marks session as having new evidence so run_dbn_inference can re-run. Valid node_name values include: stress_ema, mood, sleep_quality, sleep_disturbances, exercise, anxiety, depression, loneliness, negative_affect, positive_affect, pain_level, social_events_negative, social_events_positive, mental_stress, physical_stress, mental_health, physical_health, productivity, activity, communication.',
    inputSchema: {
      type:       'object',
      properties: {
        node_name:  { type: 'string', description: 'DBN node name (must be one of the valid names listed in the tool description).' },
        node_value: { type: 'string', description: 'Valid state label for that node.' },
        confidence: { type: 'number', description: 'Confidence [0,1].' },
        summary:    { type: 'string', description: 'One-sentence factual summary of the evidence.' },
      },
      required: ['node_name', 'node_value', 'confidence', 'summary'],
    },
    handler: (args, db, tid) => handleStoreIndirectEvidence(args, db, tid),
  },
  {
    name:        'get_cascade_questions',
    description: 'Return pending cascade questions for the current turn, based on columns already filled by NER.',
    inputSchema: {
      type:       'object',
      properties: { turn_id: { type: 'string', description: 'Current turn UUID.' } },
      required:   ['turn_id'],
    },
    handler: (args, db) => handleGetCascadeQuestions(args, db),
  },
  {
    name:        'get_belief_trend',
    description: 'Returns trend summary for specified DBN nodes over a time window. Call proactively for any nodes that changed this turn, and also when the user asks about trends, history, or changes over time.',
    inputSchema: {
      type:       'object',
      properties: {
        node_names:  { type: 'array',  items: { type: 'string' }, description: 'Node names to fetch trends for' },
        window_days: { type: 'number', description: 'Lookback window in days (default 14)' },
        session_id:  { type: 'string', description: 'Current session ID' },
      },
      required: ['node_names'],
    },
    handler: (args, db) => handleGetBeliefTrend(args, db),
  },
];

// ── Dispatcher ────────────────────────────────────────────────────────────────

export async function dispatchTool(
  name:   string,
  args:   Record<string, unknown>,
  db:     DB,
  turnId: string,
): Promise<unknown> {
  const tool = MCP_TOOLS.find(t => t.name === name);
  if (!tool) return { error: `Unknown tool: ${name}` };
  return tool.handler(args, db, turnId);
}
