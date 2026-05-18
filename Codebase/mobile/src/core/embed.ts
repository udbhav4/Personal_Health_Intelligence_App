import { initLlama, type LlamaContext } from 'llama.rn';
import type { DB } from '@op-engineering/op-sqlite';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MemorySummary {
  id:            number;
  session_id:    string;
  summary_text:  string;
  message_count: number;
  created_at:    string;
  score:         number;
}

// ── Singleton ─────────────────────────────────────────────────────────────────

let _ctx:       LlamaContext | null = null;
let _modelPath: string | null       = null;

export function setEmbedPath(modelPath: string): void {
  _modelPath = modelPath;
}

export async function initEmbed(modelPath: string): Promise<void> {
  if (_ctx && _modelPath === modelPath) return;
  if (_ctx) {
    await _ctx.release();
    _ctx = null;
  }
  _modelPath = modelPath;
  _ctx = await initLlama({
    model:          modelPath,
    embedding:      true,
    embd_normalize: 1,
    pooling_type:   'mean',    // required for nomic-embed-text-v1.5
    n_ctx:          2048,
    n_threads:      4,
    n_parallel:     1,
  });
}

export async function releaseEmbed(): Promise<void> {
  if (!_ctx) return;
  await _ctx.release();
  _ctx       = null;
  _modelPath = null;
}

export function isEmbedReady(): boolean { return _ctx !== null; }

/** Reinitialise from stored path if context was evicted by OS. */
export async function ensureEmbed(): Promise<void> {
  if (_ctx) return;
  if (!_modelPath) throw new Error('embed: initEmbed() has not been called — no model path stored');
  await initEmbed(_modelPath);
}

// ── Core embed ────────────────────────────────────────────────────────────────

export async function embedText(text: string): Promise<Float32Array> {
  await ensureEmbed();
  const result = await _ctx!.embedding(text, { embd_normalize: 1 });
  return new Float32Array(result.embedding);
}

function toBlob(arr: Float32Array): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

// ── Storage ───────────────────────────────────────────────────────────────────

export async function storeMemory(
  db:           DB,
  sessionId:    string,
  summaryText:  string,
  messageCount: number,
): Promise<void> {
  const vec  = await embedText(`search_document: ${summaryText}`);
  const blob = toBlob(vec);
  const now  = new Date().toISOString().replace('T', ' ').slice(0, 19);
  db.executeSync(
    `INSERT INTO memory_summaries (timestamp, session_id, summary_text, embedding, message_count)
     VALUES (?, ?, ?, ?, ?)`,
    [now, sessionId, summaryText, blob, messageCount],
  );
}

// ── Search with log-linear hybrid decay ───────────────────────────────────────

/**
 * FinalScore = S_vec × [1 - α × (ln(t+1) / ln(T_max+1))]
 * S_vec = 1 - vec_distance_cosine (cosine similarity from unit vectors)
 * t     = days since created_at (capped at T_max)
 * Glance: alpha=0.4, tMax=14   Reflect: alpha=0.2, tMax=90
 */
export async function searchMemory(
  db:         DB,
  queryText:  string,
  windowDays: number,
  limit:      number,
  alpha:      number = 0.4,
  tMax:       number = 14,
): Promise<MemorySummary[]> {
  const cutoff = new Date(Date.now() - windowDays * 86_400_000)
    .toISOString().replace('T', ' ').slice(0, 19);

  interface RawRow {
    id:            number;
    session_id:    string;
    summary_text:  string;
    message_count: number;
    created_at:    string;
    distance:      number;
  }

  interface RecencyRow {
    id:            number;
    session_id:    string;
    summary_text:  string;
    message_count: number;
    created_at:    string;
  }

  let rows: RawRow[] = [];
  let useRecencyFallback = false;

  try {
    const queryVec  = await embedText(`search_query: ${queryText}`);
    const queryBlob = toBlob(queryVec);

    rows = db.executeSync(
      `SELECT id, session_id, summary_text, message_count, created_at,
              vec_distance_cosine(embedding, ?) AS distance
       FROM   memory_summaries
       WHERE  embedding IS NOT NULL
         AND  created_at >= ?
       ORDER  BY distance ASC
       LIMIT  ?`,
      [queryBlob, cutoff, limit * 4],
    ).rows as unknown as RawRow[];

    if (rows.length === 0) {
      useRecencyFallback = true;
    }
  } catch {
    useRecencyFallback = true;
  }

  if (useRecencyFallback) {
    const recencyRows = db.executeSync(
      `SELECT id, session_id, summary_text, message_count, created_at
       FROM memory_summaries
       WHERE created_at >= ?
       ORDER BY created_at DESC
       LIMIT ?`,
      [cutoff, limit],
    ).rows as unknown as RecencyRow[];

    return recencyRows.map(r => ({
      id:            r.id,
      session_id:    r.session_id,
      summary_text:  r.summary_text,
      message_count: r.message_count,
      created_at:    r.created_at,
      score:         1.0,
    }));
  }

  const now    = Date.now();
  const lnTMax = Math.log(tMax + 1);

  return rows
    .map(r => {
      const sVec    = 1 - Number(r.distance) / 2;   // cosine dist ∈ [0,2] → sim ∈ [0,1]
      const tDays   = (now - new Date(r.created_at).getTime()) / 86_400_000;
      const tCapped = Math.min(tDays, tMax);
      const score   = sVec * (1 - alpha * (Math.log(tCapped + 1) / lnTMax));
      return {
        id:            r.id,
        session_id:    r.session_id,
        summary_text:  r.summary_text,
        message_count: r.message_count,
        created_at:    r.created_at,
        score,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
