# Backend Fix Plan — Full Issue Registry

---

## CRITICAL

### C1 — `inter_trans` 1D array crashes inference from 2nd run onwards
**File:** `Codebase/backend/export_cpds.py`, `core/inferenceEngine.ts` `applyInterSlice()`
**Problem:** `export_cpds.py` writes `inter_trans[node]` as a flat 1D array for static/background
nodes (age, sex, etc. — no t=1 parent). `applyInterSlice` does `trans.map(row => row.reduce(...))`
— expects `row = number[]`, gets `row = number` → `TypeError: row.reduce is not a function`.
Crashes entire inference from the 2nd run onward (first run has no prior snapshot, so safe).
**Fix:** In `export_cpds.py`, always reshape `vals` to 2D before `.tolist()`. If result is 1D
after averaging, wrap as `[vals]` (identity-like transition matrix).
Then re-run `export_cpds.py` + `sync_configs.py`.

---

### C2 — `session_id` column does not exist → every Talk turn fails
**File:** `core/agent.ts` ~line 192
**Problem:** SQL query references `session_id` column in `user_data_sensorless`. Column does not
exist in schema (`db.ts`). SQLite throws `no such column: session_id` on every `startTurn()` call.
No try/catch → entire Talk flow dead.
**Fix:** Remove or replace `session_id` filter with the correct column from schema.

---

### C3 — `source_column_bins.state_labels` mismatch with node `state_labels`
**File:** `configs/feature_node_config.json`, `core/nlu.ts` `resolveNodeValue()`
**Problem:** `resolveNodeValue()` returns labels from `source_column_bins.state_labels`.
These don't match node-level `state_labels` (CPD ground truth) for 8+ nodes.
`inferenceEngine` does `slist.indexOf(node_value)` → returns -1 → pins node to state[0] always.
Affected: smoking, sleep_quality, exercise, mood, productivity, social_events_positive/negative.
**Fix (two parts):**
1. Fix `feature_node_config.json`: for every `source_column_bins` entry, `state_labels` must
   match the parent node's `state_labels` exactly (CPDs are authoritative).
2. Add guard in `resolveNodeValue()`: validate returned label exists in node's `state_labels`,
   return `null` if not.
3. For columns where source has MORE states than node (e.g. 4 col states → 3 node states via
   k-modes): add `state_map` field in `source_column_bins` entry mapping col label → node label.
   `resolveNodeValue()` applies this map before writing `node_value`.
Then run `sync_configs.py`.

---

### C4 — `startTurn()` crashes if embed model not yet initialized
**File:** `core/agent.ts` `startTurn()`, `core/embed.ts` `embedText()`
**Problem:** `startTurn()` calls `dispatchTool('get_user_memory')` → `searchMemory()` →
`embedText()`. If `initEmbed()` not complete, throws `"embed: not initialized"`. No try/catch
in dispatch path → `startTurn()` rejects. Affects any Talk turn during startup race.
**Fix:** See Model Readiness section below.

---

## HIGH

### H5 — Undo baseline overwritten by `handleRunDbnInference`
**File:** `core/agent.ts` `runUndoWork()`, `core/mcp.ts` `handleRunDbnInference()`
**Problem:** `runUndoWork()` calls `overrideTurnStart(preUndoBeliefs)` to pin the
`get_changed_nodes` diff baseline. Immediately after, `handleRunDbnInference()` fires and resets
`turnStartBeliefs = latestBeliefs` (which still reflects erroneous pre-undo data), overwriting
the override. Undo diff is computed from wrong baseline → Gemma gets misleading context.
**Fix:** In `handleRunDbnInference()`, skip the `turnStartBeliefs` reset if an override is
currently active (add `_turnStartOverrideActive` flag, cleared after first `get_changed_nodes` call).

---

### H6 — `passthrough` columns always write `node_value = null`
**File:** `core/nlu.ts` `resolveNodeValue()`
**Problem:** `resolveNodeValue()` only handles `method: clinical`. All `passthrough` columns
(sex, marital_status, general_health, diabetes_status, chronic_condition, pain_level, smoking,
exercise, time_of_day) return `null`. `fetchNodeRows()` filters `WHERE node_value IS NOT NULL`
→ these rows never reach DBN inference.
**Fix:** Add `passthrough` branch in `resolveNodeValue()`: raw string value IS the node state,
write directly as `node_value` with no discretization.

---

### H7 — Few-shot examples use invalid node names, silently dropped
**File:** `core/nlu.ts` `CLASSIFY_INTENT_SHOTS` / `RESPONSE_GUIDE`
**Problem:** Few-shot examples reference `mood_valence` and `sleep_duration_hrs` as `queryNodes`.
Neither is a valid node (actual: `mood`, `sleep_quality`). `validNodes.has(n)` filter drops them.
`hasTrendQuery` may be true but `queryNodes` is empty → `get_belief_trend` returns `{}` →
Gemma receives no trend data.
**Fix:** Replace `mood_valence` → `mood`, `sleep_duration_hrs` → `sleep_quality` in all
few-shot examples and response guide strings.

---

### H8 — `writeProactiveAnswer()` writes `node_value = null` → cascade answers never reach DBN
**File:** `core/agent.ts` `writeProactiveAnswer()`
**Problem:** Cascade question answers inserted with `node_value` omitted (NULL in SQLite).
`evidenceLayer.fetchNodeRows()` filters `WHERE node_value IS NOT NULL` → answers silently
discarded → cascade questions functionally useless for inference.
**Fix:** Call `resolveNodeValue()` before inserting (same as `runNer()` does for L1 entities).

---

### H9 — `computePrevDayAggregates` uses old activity threshold 0.33 (updated to 0.25)
**File:** `core/passiveSensing.ts` line ~142
**Problem:** Re-discretization of activity for prev_day uses `< 0.33` threshold.
`activityCollector.ts` was updated to `CLINICAL_THRESHOLD = 0.25`. Mismatch in 0.25–0.33 range
produces wrong `prev_day_activity` discretized_value.
**Fix:** Change `< 0.33` → `<= 0.25` to match `activityCollector.ts` `CLINICAL_THRESHOLD`
(note: `clinicalBin` uses `<=` not `<`).

---

## MEDIUM

### M10 — UTC midnight timezone bug in `toExpiresDate()`
**File:** `core/nlu.ts` ~line 551
**Problem:** `new Date('YYYY-MM-DD')` parses as UTC midnight. In UTC-minus timezones, resolves
to previous local day → `expiresDate` 1 day early → entity suppression wears off too soon.
**Fix:** Parse as local midnight: replace `new Date(baseDate)` with
`new Date(baseDate + 'T00:00:00')` (no trailing Z).

---

### M11 — `isSuppressed()` missing `ORDER BY` → may return stale suppression
**File:** `core/questionCascade.ts` ~line 55
**Problem:** `SELECT ... LIMIT 1` with no `ORDER BY created_at DESC`. SQLite may return the
oldest row. If oldest has expired suppression and newest has valid one → suppression not detected
→ user re-asked a recently answered question.
**Fix:** Add `ORDER BY created_at DESC` before `LIMIT 1`.

---

### M12 — `store_indirect_evidence` does not validate `node_name` against DBN graph
**File:** `core/mcp.ts` `handleStoreIndirectEvidence()` ~line 321
**Problem:** Only checks `!nodeName || !nodeValue`. Gemma can pass any string (e.g.
`"fatigue_level"`) not in the DBN. Row written to DB but never consumed by evidenceLayer.
Silent data loss.
**Fix:** Validate `nodeName` exists in `MODEL_STATES` (or `STALENESS_DAYS`). Return tool error
if invalid, with list of valid node names for Gemma to retry.

---

### M13 — `_pendingTurns` leaks if user abandons question prompt
**File:** `core/agent.ts` ~line 793
**Problem:** `_pendingTurns.set(turnId, {...})` written when `done: false`. If user dismisses
question screen or app backgrounds without calling `completeTurn()`, entry leaks indefinitely.
**Fix:** Add TTL eviction: on each `startTurn()` call, evict any pending entries older than
30 minutes. Also clear on session reset.

---

### M14 — Mixed UTC/local timestamps between sensor_windows and inference_snapshots
**File:** `core/passiveSensing.ts`, `core/mcp.ts` `handleRunDbnInference()`, `core/infer.ts`
**Problem:** Sensor windows use device-local timestamps (`'sv'` locale → local date).
Inference snapshots use UTC (`toISOString()`). Cross-table ordering queries can produce
incorrect day-boundary behavior in UTC+ timezones.
**Fix:** Standardize all timestamps to device-local ISO strings using `'sv'` locale
(`new Date().toLocaleString('sv').replace(' ', 'T')`). Apply consistently across
`passiveSensing.ts`, `mcp.ts`, `infer.ts`, `db.ts`.

---

### M15 — `prev_day_*` nodes and sensor nodes absent from `STALENESS_DAYS`
**File:** `core/evidenceLayer.ts`
**Problem:** `buildDbnEvidence()` only iterates `Object.keys(STALENESS_DAYS)` for sensorless
nodes. `prev_day_*` nodes are not in `STALENESS_DAYS` → never read via `fetchNodeRows`.
Sensor nodes only read via `fetchSensorSnapshot()` (latest window only) — if no recent window,
node silently absent from evidence AND from prior_factors.
**Fix:** See Evidence Completeness section below.

---

## LOW

### L16 — `_note` metadata key included in L2 NER prompt
**File:** `core/nlu.ts` L2 prompt construction ~line 154
**Problem:** `source-column-descriptions.json` has a `_note` key (schema comment).
Included in L2 prompt as if it were a valid column. Minor prompt noise.
**Fix:** Filter out keys starting with `_` when building L2 prompt.

---

### L17 — `session_id` required in `get_belief_trend` schema but ignored in handler
**File:** `core/mcp.ts` `handleGetBeliefTrend()` ~line 498
**Problem:** Tool schema marks `session_id` as required but handler never uses it.
Misleading for any future multi-user extension.
**Fix:** Either remove `session_id` from schema, or use it in the query.

---

### L18 — Co-influenced sibling nodes mislabeled as `effect`
**File:** `core/mcp.ts` `handleGetChangedNodes()` ~line 262
**Problem:** Nodes sharing a parent with an evidence node (co-influenced siblings) are labeled
`'effect'`. They are neither causes nor strict effects. Gemma may misinterpret the relationship.
**Fix:** Add `'co-influenced'` label for nodes that share a parent with an evidence node
but are not in `evidenceDescendants`.

---

### L19 — `isPureUndo` bug: pure text correction gets no undo
**File:** `core/agent.ts` `runUndoWork()` ~line 605
**Problem:** If `nerNodes.length === 0 && nerResult.unmatched.length > 0`, returns
`{ preUndoBeliefs: null, isPureUndo: false }` without calling `dispatchTool('undo_last_entry')`.
User saying "Actually disregard that" gets no undo.
**Fix:** Detect pure correction intent via `detectUndoIntent()` result. If undo intent confirmed
and NER empty → still call `dispatchTool('undo_last_entry')`, set `isPureUndo: true`.

---

### L20 — L1 dedup loses `report_scope` when two segments match same `original_column`
**File:** `core/nlu.ts` `deduplicateEntities()` ~line 541
**Problem:** Dedup key is `original_column`. Two segments matching same column (e.g.
"I felt sad today" + "I've been sad this week") deduplicate to higher-confidence one.
The `report_scope` difference (today vs week) is lost.
**Fix:** Include `report_scope` in dedup key: `${original_column}||${report_scope}`.

---

### L21 — sqlite-vec extension load not validated at startup
**File:** `core/embed.ts` ~line 121, `core/db.ts` ~line 17
**Problem:** If native build does not include sqlite-vec, `vec_distance_cosine` throws an
opaque SQL error on first `searchMemory()` call. No JS-side check that extension loaded.
**Fix:** In `initEmbed()` or `db.ts` init, run a probe query (`SELECT vec_distance_cosine(...)`)
and throw a descriptive error early if extension is missing.

---

## Model Readiness — No Gaps in Agentic Flow (expanded from C4)

### Problem Summary

| Model | Current not-ready behavior | Risk |
|-------|---------------------------|------|
| NLU (`nlu.ts`) | Silent skip → returns empty | Evidence never written, inference runs blind |
| Embed (`embed.ts`) | Throws | `startTurn` crashes |
| Agent (`agent.ts`) | Throws | `startTurn` crashes |
| OS eviction | `_ctx` non-null but invalid | Native error, unhandled |

### Step 1 — Eager init at app startup
New file `core/initModels.ts`:
- Calls `initNlu(path)` → `initEmbed(path)` → `initAgent(path)` in sequence
- Returns a promise UI splash screen awaits before navigating to app
- Stores each `modelPath` in module-level variable for reinit on eviction

### Step 2 — Add `ensureReady()` pattern to each model module
Each module gets `_modelPath: string` stored at init time and a new `ensure*()` async function:
```
ensureNlu() / ensureEmbed() / ensureAgent():
  if _ctx === null → reinit from stored _modelPath
```

### Step 3 — Wrap all model calls with eviction catch + single retry
```
try {
  result = await _ctx.completion(...)
} catch (e) {
  if (isNativeContextError(e)) {
    _ctx = null
    await ensure*()               // reinit once from stored path
    result = await _ctx.completion(...)  // retry
  } else throw
}
```

### Step 4 — Replace all silent-skip guards in `nlu.ts`
All 4 locations (`segmentText`, `runNer`, `classifyIntent`, `detectUndoIntent`):
`if (!_ctx) return empty` → `await ensureNlu()` then proceed normally

### Step 5 — Add `isEmbedReady()` export to `embed.ts`
Mirrors `isNluReady()` / `isAgentReady()` already present in other modules.

### Files to Touch
| File | Change |
|------|--------|
| `core/initModels.ts` | New — eager init orchestrator |
| `core/nlu.ts` | Replace 4 skip guards with `ensureNlu()` |
| `core/embed.ts` | Add `_modelPath`, `ensureEmbed()`, `isEmbedReady()` |
| `core/agent.ts` | Add `_modelPath`, `ensureAgent()` with eviction retry |

---

## Evidence Completeness — All Data Points Reach DBN

**Direction:** Every node always contributes via collected data or marginal priors. No node
silently absent from LBP.

### Gap Audit

| Data path | Reaches DBN? | Gap |
|-----------|-------------|-----|
| Sensorless nodes in `STALENESS_DAYS` | Yes | OK |
| Sensorless nodes NOT in `STALENESS_DAYS` | No | BAD |
| `prev_day_*` nodes | No | BAD |
| Sensor nodes (latest window present) | Yes | OK |
| Sensor nodes (no recent window) | No | BAD |
| `passthrough` cols | No (null node_value) | BAD — fixed by H6 |
| Cascade answers | No (null node_value) | BAD — fixed by H8 |
| Any node with no data at all | Falls to marginal priors | OK (trained priors) |

### Fix 1 — Add `prev_day_*` nodes to evidence path
`prev_day_*` rows are written to `sensor_windows`, not `user_data_sensorless`.
`buildDbnEvidence` must read them separately from `sensor_windows` (same as sensor nodes)
with staleness = 1 day. Add dedicated `fetchPrevDaySnapshot(db)` call in `buildDbnEvidence`.

### Fix 2 — Sensor nodes with no recent window → inject marginal priors
After `fetchSensorSnapshot()`, for any DBN sensor node with no snapshot in the result,
inject `MARGINAL_PRIORS[node]` as `prior_factors[node]`. Node participates in LBP via priors.

### Fix 3 — Universal marginal prior fallback for all absent nodes
After all evidence/prior_factor collection, iterate ALL nodes in `model.states`
(from `cpd_tables.json` — the complete DBN node list). For any node absent from both
`evidence` and `prior_factors`, inject `MARGINAL_PRIORS[node]` as `prior_factors[node]`.
Guarantees cold-start inference (first open, no data) uses trained marginal priors, not
implicit LBP uniform assumption.

### Files to Touch
| File | Change |
|------|--------|
| `core/evidenceLayer.ts` | Add `fetchPrevDaySnapshot()`, sensor marginal fallback, universal fallback |

---

## L2 Composite Instruments — Partial Score Resolution

Currently: individual items stored with `node_value = null` until all items answered.
Partial data never influences beliefs even when mostly complete.

### Fix — Partial confidence threshold
- After each new item write, count answered items for the composite in DB
- answered / total < 0.8 → skip, keep `node_value = null` (too unreliable)
- answered / total >= 0.8 → compute partial score → discretize → write `node_value`
  with scaled confidence: `0.65 + 0.25 * (answered / total)`
  - e.g. 4/5 → confidence 0.85; 5/5 → 0.90 (then standard history logic applies)

### Files to Touch
| File | Change |
|------|--------|
| `core/agent.ts` | Add composite completeness check in `writeProactiveAnswer()` |
| `core/nlu.ts` | Add composite completeness check after L2 item insert in `runNer()` |

---

## Post-Training Export

After every training run, execute in order:
1. `python -m Codebase.backend.export_cpds` — reads `models/dbn_model.pkl` → `configs/cpd_tables.json`
2. `python -m Codebase.backend.export_column_question_map` — only if `column_question_map.csv` changed
3. `python Codebase/backend/sync_configs.py` — copies all configs → mobile assets + regenerates `populationNormStats.ts`
